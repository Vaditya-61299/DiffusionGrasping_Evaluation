### select the model ###

## FOR SE3DIF USE THIS
c=1
input="grasp_dif_multi"

## FOR CGDF USE THIS
#c=2
#input="cgdf_v1"

grasp_gen_mode=False
eval_model=True

### functions here ###

def parse_args(input):
    p = configargparse.ArgumentParser()
    p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

    p.add_argument('--obj_id', type=int, default=0)             # if we run a loop to get all the objects of a class, then the object id will be required to change
    p.add_argument('--n_grasps', type=str, default='50')       # #### TO DO: Need to look out for that for CGDF as it generates a random number of samples and it might not coincide with the number of grasps we have asked for.
    p.add_argument('--obj_class', type=str, default='Mug')      # change this for a different object
    p.add_argument('--device', type=str, default='cuda:0')      # uses gpu.. if no gpu then change to 'cpu'
    p.add_argument('--eval_sim', type=bool, default=False)      # False when no need to evaluate the generted grasps.
    p.add_argument('--model', type=str, default=input)          # just running one model at a time. make sure first that individual models are working properly


    opt = p.parse_args()
    return opt


def get_model_and_generator(p, args, device='cpu',model_input=1):
    model_params = args.model
    batch = int(args.n_grasps)
    ## Load model
    model_args = {
        'device': device,
        'pretrained_model': model_params
    }
    model = load_model(model_args)

    context = to_torch(p[None,...], device)
    model.set_latent(context, batch=batch)

    ########### 2. SET SAMPLING METHOD #############
    if model_input==1 or model_input==2:
        generator = Grasp_AnnealedLD(model, batch=batch, T=70, T_fit=50, k_steps=2, device=device, model_input=c,)          ### model c tells which generator.. for se3dif or cgdf (for now)... #### TO DO: CHANGE THE CONDITION SO THAT GraspLDM IS ALSO ACCEPTED
        return generator, model

    if model_input==3:
        print("Model not available for this number(graspLDM)")
        raise RuntimeError
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
    if model_input==2:
        mesh= mean_center_and_normalize(mesh)
    P = mesh.sample(1000)

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

import torch

###     argument parsing here and set up the code for logging ###
#from isaac_evaluation.utils.log_utils import log_dir
#log_dir()

args = parse_args(input)

print('##########################################################')
print('Object Class: {}'.format(args.obj_class))
print(args.obj_id)
print('Model: {}'.format(args.model))
print('##########################################################')

n_grasps = int(args.n_grasps)
obj_id = int(args.obj_id)
obj_class = args.obj_class
n_envs = 10         #TODO: CHANGE TO 1 since it delivers best results
device = args.device

## Set Model and Sample Generator ##
P, mesh, trans, rot_quad = sample_pointcloud(obj_id, obj_class,model_input=c)

generator, model = get_model_and_generator(P, args, device,model_input=c)
print ("Code working fine for now till here")



if grasp_gen_mode and eval_model==False:
    print("Grasp generation taking place. Generating", n_grasps, "grasps")
    if c==1:
        H = generator.sample(model_input=c)           #t needed for cgfd      see if this causes problem for the se3dif           t is needed just for energy based grasps
    if c==2:
        H  ,t = generator.sample(model_input=c)
    H_grasp = copy.deepcopy(H)
    H_grasp[:, :3, -1] = (H_grasp[:, :3, -1] - torch.as_tensor(trans[:3,-1],device=device)).float()
    H[..., :3, -1] *=1/8.
    H_grasp[..., :3, -1] *=1/8.
    print(H)

    print("no problem till here")
    EVAL_SIMULATION = args.eval_sim
        # isaac gym has to be imported here as it is supposed to be imported before torch
    if (EVAL_SIMULATION):
            # Alternatively: Evaluate Grasps in Simulation:
        from evaluation.grasp_quality_evaluation import GraspSuccessEvaluator
        print("in evaluation mode now")
        ## Evaluate Grasps in Simulation##
#        num_eval_envs = 10  # was 10
        evaluator = GraspSuccessEvaluator(obj_class, n_envs=n_envs, idxs=[args.obj_id] * n_envs, viewer=False, device=device, \
                                          rotations=[rot_quad]*n_envs, enable_rel_trafo=False)
        succes_rate = evaluator.eval_set_of_grasps(H_grasp)
        print('Success cases : {}'.format(succes_rate))



elif eval_model and grasp_gen_mode==False:
    from evaluation.grasp_quality_evaluation.evaluate_model import EvaluatePointConditionedGeneratedGrasps
    evaluator = EvaluatePointConditionedGeneratedGrasps(generator, n_grasps=n_grasps, obj_id=obj_id, obj_class=obj_class, n_envs=n_envs,
                                                        viewer=False,model_input=c)

    robust_cases,success_cases, edd_mean, edd_std = evaluator.generate_and_evaluate(success_eval=True, earth_moving_distance=True)
    #print('Success cases : {}'.format(success_cases))
    print(f"earth moving distance- dataset-datset:       mean: {edd_mean}       standard deviation: {edd_std}")



elif eval_model and grasp_gen_mode:
    print("TOO MUCH COMPUTATION. SELECT ONLY ONE MODEL")

else:
    print("NO EVAULATION")