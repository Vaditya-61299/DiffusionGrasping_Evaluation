### select the model ###

## FOR SE3DIF USE THIS
c=1
input="grasp_dif_multi"

## FOR CGDF USE THIS
#c=2
#input="cgdf_v1"

## FOR GRASP_LDM USE THIS
#c=3
#input="graspLDM"

grasp_gen_mode=False
eval_model=True

object_classes=["Mug","CerealBox","WineBottle","Plant","Knife","PSP","Spoon","Camera","WineGlass","Teacup","ToyFigure","Hat","Book","Ring","Hammer"]
#object_classes=["Mug"]
def parse_args(input):
    p = configargparse.ArgumentParser()
    p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

    p.add_argument('--obj_id', type=int, default=0)             # if we run a loop to get all the objects of a class, then the object id will be required to change
    p.add_argument('--n_grasps', type=str, default='50')       # #### TO DO: Need to look out for that for CGDF as it generates a random number of samples and it might not coincide with the number of grasps we have asked for.
    p.add_argument('--obj_class', type=str, default='Mug')      # change this for a different object
    p.add_argument('--device', type=str, default='cuda:0')      # uses gpu.. if no gpu then change to 'cpu'
    p.add_argument('--eval_sim', type=bool, default=False)      # False when no need to evaluate the generted grasps.
    p.add_argument('--model', type=str, default=input)          # just running one model at a time. make sure first that individual models are working properly

    if input=="graspLDM":
        data_root=os.path.join(os.path.join(os.path.dirname(__file__)),"data")
        exp_path=os.path.join(data_root,"models/GraspLDM/checkpoints/generation/fpc_1a_latentc3_z4_pc64")
        p.add_argument("--exp_path", type=str, default=exp_path ,help="Path to experiment checkpoint")
        p.add_argument("--data_root", type=str, default=data_root, help="Root directory for data")
        p.add_argument(
        "--no_ema",
        action="store_false",
        dest="use_ema_model",
        help="Disable EMA model usage",
    )
        p.add_argument("--split", type=str, default="test", help="Data split to use")
        p.add_argument(
        "--inference_steps",
        type=int,
        default=100,
        help="Number of inference steps for LDM",
    )

    opt = p.parse_args()
    return opt

def set_graspLDM_model(obj_class):
    exp_name = os.path.basename(args.exp_path)
    exp_out_root = os.path.dirname(args.exp_path)
    model = InferenceLDM(
            exp_name=exp_name,
            exp_out_root=exp_out_root,
            use_elucidated=False,
            data_root=args.data_root,
            load_dataset=True,
            num_inference_steps=args.inference_steps,
            use_fast_sampler=False,
            data_split=args.split,
            use_ema_model=args.use_ema_model,
            obj_class=obj_class,
        )
    return model
def get_model_and_generator(p, args, device='cpu',model_input=1):
    model_params = args.model
    batch = int(args.n_grasps)
    ## Load model
    model_args = {
        'device': device,
        'pretrained_model': model_params
    }
    context = to_torch(p[None,...], device)
    ########### 2. Set Model #############
    if model_input==1 or model_input==2:
        model = load_model(model_args)
        
        model.set_latent(context, batch=batch)
        generator = Grasp_AnnealedLD(model, batch=batch, T=70, T_fit=50, k_steps=2, device=device, model_input=c,)          ### model c tells which generator.. for se3dif or cgdf (for now)... #### TO DO: CHANGE THE CONDITION SO THAT GraspLDM IS ALSO ACCEPTED
        return generator, model

    else:
        print("Model not available for this number")
        raise RuntimeError

def mean_center_and_normalize(mesh):
    mean = mesh.sample(1000).mean(0)
    mesh.vertices -= mean
    scale = np.max(np.linalg.norm(mesh.vertices, axis=1))
    mesh.apply_scale(1/(2*scale))
    return mesh

def sample_pointcloud(obj_id=0, obj_class='Gun',model_input=1):
    acronym_grasps = AcronymGraspsDirectory(data_type=obj_class)
    mesh = acronym_grasps.avail_obj[obj_id].load_mesh()
#    if model_input==2:
#        mesh= mean_center_and_normalize(mesh)
    P = mesh.sample(1024)

    sampled_rot = scipy.spatial.transform.Rotation.random()
    rot = sampled_rot.as_matrix()

    rot_quat = sampled_rot.as_quat()            ###only se3dif  for eval_sim mode

    P = np.einsum('mn,bn->bm', rot, P)
    P *= 8.
    P_mean = np.mean(P, 0)
    P += -P_mean

    H = np.eye(4)
    H[:3,:3] = rot
    mesh.apply_transform(H)
    mesh.apply_scale(8.)
    H = np.eye(4)
    H[:3,-1] = -P_mean
    mesh.apply_transform(H)

    translational_shift = copy.deepcopy(H)

    return P, mesh, translational_shift, rot_quat

###         import Modules here ###     ## IMPORTANT: ALWAYS IMPORT BELOW ISAACGYM TO AVOID IMPORT CONFLICTS#
import isaacgym    
from isaacgym import gymutil,gymapi

import copy
import configargparse

import scipy.spatial.transform

import numpy as np

from DiffusionFields.se3dif.datasets import AcronymGraspsDirectory
from DiffusionFields.se3dif.models.loader import load_model
from DiffusionFields.se3dif.samplers import ApproximatedGrasp_AnnealedLD, Grasp_AnnealedLD
from DiffusionFields.se3dif.utils import to_numpy, to_torch

from graspLDM.tools.inference import Conditioning, InferenceLDM, InferenceVAE, ModelType
from graspLDM.grasp_ldm.utils.rotations import tmrp_to_H

import os

import torch


args = parse_args(input)

print('##########################################################')
print('Object Class: {}'.format(args.obj_class))
print(args.obj_id)
print('Model: {}'.format(args.model))
print('##########################################################')

n_grasps = int(args.n_grasps)
obj_id = int(args.obj_id)
obj_class = args.obj_class
n_envs = 1         #TODO: CHANGE TO 1 since it delivers best results
device = args.device

## Set Model and Sample Generator ##
P, mesh, trans, rot_quad = sample_pointcloud(obj_id, obj_class,model_input=c)

if c==1 or c==2:
    generator, model = get_model_and_generator(P, args, device,model_input=c)
if c==3:
    pass
print ("Code working fine for now till here")



if grasp_gen_mode and eval_model==False:
    print("Grasp generation taking place. Generating", n_grasps, "grasps")
    if c==1:
        H = generator.sample(model_input=c)           #t needed for cgfd      see if this causes problem for the se3dif           t is needed just for energy based grasps
        H_grasp = copy.deepcopy(H)
        H_grasp[:, :3, -1] = (H_grasp[:, :3, -1] - torch.as_tensor(trans[:3,-1],device=device)).float()
        H[..., :3, -1] *=1/8.
        H_grasp[..., :3, -1] *=1/8.
    if c==2:
        H  ,t = generator.sample(model_input=c)
        H_grasp = copy.deepcopy(H)
        H_grasp[:, :3, -1] = (H_grasp[:, :3, -1] - torch.as_tensor(trans[:3,-1],device=device)).float()
        H[..., :3, -1] *=1/8.
        H_grasp[..., :3, -1] *=1/8.
    if c==3:
        model=set_graspLDM_model(args.obj_class)
        condition_type = Conditioning.UNCONDITIONAL
        conditioning = None
        results = model.infer(
         data_idx=args.obj_id,
         num_grasps=args.num_grasps,
         visualize=args.visualize,
         condition_type=condition_type,
         conditioning=conditioning,
        )
        H_grasps=results["grasps"]
        H=torch.squeeze(H_grasps)

    print("no problem till here")
    EVAL_SIMULATION = args.eval_sim
        # isaac gym has to be imported here as it is supposed to be imported before torch
    if (EVAL_SIMULATION):
            # Alternatively: Evaluate Grasps in Simulation:
        from evaluation.grasp_quality_evaluation import GraspSuccessEvaluator
        print("in evaluation mode now")
        ## Evaluate Grasps in Simulation##
#        num_eval_envs = 10  # was 10
        evaluator = GraspSuccessEvaluator(obj_class, n_envs=n_envs, idxs=[args.obj_id] * n_envs, viewer=True, device=device, \
                                          rotations=[rot_quad]*n_envs, enable_rel_trafo=False)
        succes_rate = evaluator.eval_set_of_grasps(H_grasp)
        print('Success cases : {}'.format(succes_rate))



elif eval_model and grasp_gen_mode==False:
    len=len(AcronymGraspsDirectory(data_type=args.obj_class).avail_obj)
    from evaluation.grasp_quality_evaluation.evaluate_model import EvaluatePointConditionedGeneratedGrasps
    if c==2 or c==1:
        for j in object_classes:
            obj_class=j
            
            for i in range(0,5):
                obj_id=i
                if obj_class=="Camera" and obj_id==4:   # id 4 for camera has no good grasps and we need that for EMD
                    obj_id+=1
        
                evaluator = EvaluatePointConditionedGeneratedGrasps(generator, n_grasps=n_grasps, obj_id=obj_id, obj_class=obj_class, n_envs=n_envs,
                                                            viewer=False,model_input=c)

                success_cases, edd_mean, edd_std = evaluator.generate_and_evaluate(success_eval=True, earth_moving_distance=True)
        #print('Success cases : {}'.format(success_cases))
                print(f"earth moving distance- dataset-datset:       mean: {edd_mean}       standard deviation: {edd_std}")
                
                del evaluator
    if c==3:
        for j in object_classes:
            obj_class=j
            generator=set_graspLDM_model(obj_class)

            for i in range(0,5):
                obj_id=i
                if obj_class=="Camera" and obj_id==4:   # id 4 for camera has no good grasps and we need that for EMD
                    obj_id+=1
        
                evaluator = EvaluatePointConditionedGeneratedGrasps(generator, n_grasps=n_grasps, obj_id=obj_id, obj_class=obj_class, n_envs=n_envs,
                                                            viewer=False,model_input=c)

                success_cases, edd_mean, edd_std = evaluator.generate_and_evaluate(success_eval=True, earth_moving_distance=True)
        #print('Success cases : {}'.format(success_cases))
                print(f"earth moving distance- dataset-datset:       mean: {edd_mean}       standard deviation: {edd_std}")
                
                del evaluator




elif eval_model and grasp_gen_mode:
    print("TOO MUCH COMPUTATION. SELECT ONLY ONE MODEL")

else:
    print("NO EVAULATION")
