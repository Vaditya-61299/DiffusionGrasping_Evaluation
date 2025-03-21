import argparse
import os
import sys
from typing import Optional, Tuple
import isaacgym
from mesh_to_sdf.surface_point_cloud import get_scan_view, get_hq_scan_view
from mesh_to_sdf.scan import ScanPointcloud
import torch
###Usable grasp categories
###["Cup", "Mug", "Bowl", "Laptop", "Bag", "Banana", "Camera", "Hammer", "PowerSocket", "PowerStrip", "PS3", "PSP", "Ring", "Shampoo", "SodaCan", "ToiletPaper", "ToyFigure", "Wallet", "Pizza", "RubiksCube", "Tank", "USBStick"]      #works because these splits are avilabe

os.environ["LIBGL_ALWAYS_INDIRECT"] = "0"
sys.path.append((os.getcwd()))

#import isaacgym
#from isaac_evaluation.grasp_quality_evaluation import GraspSuccessEvaluator


import numpy as np
from graspLDM.tools.inference import Conditioning, InferenceLDM, InferenceVAE, ModelType

Eval=True

def parse_args(exp_path,data_root):
    parser = argparse.ArgumentParser(description="Grasp Generation Script")
    parser.add_argument(
        "--exp_path", type=str, default=exp_path ,help="Path to experiment checkpoint"
    )
    parser.add_argument(
        "--data_root", type=str, default=data_root, help="Root directory for data"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["VAE", "LDM"],
        default="LDM",
        help="Model type to use",
    )
    parser.add_argument("--split", type=str, default="test", help="Data split to use")
    parser.add_argument(
        "--num_grasps", type=int, default=20, help="Number of grasps to generate"                       # IMPORTANT
    )
    parser.add_argument("--visualize", default=False,action="store_true", help="Enable visualization")
    parser.add_argument(
        "--no_ema",
        action="store_false",
        dest="use_ema_model",
        help="Disable EMA model usage",
    )
    parser.add_argument(
        "--num_samples", type=int, default=1, help="Number of samples to generate"
    )
    parser.add_argument(
        "--conditioning",
        type=str,
        choices=["unconditional", "class", "region"],
        default="unconditional",
        help="Type of conditioning to use",
    )
    parser.add_argument(
        "--condition_value",
        type=int,
        help="Value for conditioning (class label or region ID)",
    )
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=100,
        help="Number of inference steps for LDM",
    )
    parser.add_argument(
        "--obj_class",
        type=str,
        default="Cup",
        help="Choose the object type/class",
    )
    parser.add_argument(
        "--obj_id",
        type=int,
        default=0,
        help="Index of object to use"
    )

    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Number of environment for isaac gym simulation"
    )
    return parser.parse_args()


def setup_model(args):
    
    exp_name = os.path.basename(args.exp_path)
    exp_out_root = os.path.dirname(args.exp_path)


    if args.mode == "LDM":
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
            obj_class=args.obj_class,
        )
        print(
            f"Trained using noise schedule: beta0 = {model.model.diffusion_model.beta_start} ; betaT = {model.model.diffusion_model.beta_end}"
        )
    elif args.mode == "VAE":
        model = InferenceVAE(
            exp_name=exp_name,
            exp_out_root=exp_out_root,
            data_root=args.data_root,
            load_dataset=True,
            data_split=args.split,
            use_ema_model=args.use_ema_model,
        )
    return model


def get_conditioning(args) -> Tuple[Optional[Conditioning], Optional[int]]:
    if args.conditioning == "unconditional":
        return Conditioning.UNCONDITIONAL, None
    elif args.conditioning == "class":
        if args.condition_value is None:
            raise ValueError("Must provide --condition_value for class conditioning")
        return Conditioning.CLASS_CONDITIONED, args.condition_value
    elif args.conditioning == "region":
        if args.condition_value is None:
            raise ValueError("Must provide --condition_value for region conditioning")
        return Conditioning.REGION_CONDITIONED, args.condition_value
    return None, None


def main():
    data_root=os.path.join(os.path.dirname(os.path.dirname(os.path.join(os.path.dirname(__file__)))),"data")
    exp_path=os.path.join(data_root,"models/GraspLDM/checkpoints/generation/fpc_1a_latentc3_z4_pc64")
    args = parse_args(exp_path=exp_path,data_root=data_root)
    model = setup_model(args)
    condition_type, conditioning = get_conditioning(args)

    #for _ in range(args.num_samples):
    data_idx = args.obj_id

    # Skip conditioning for VAE mode
    if args.mode == "VAE":
        condition_type = Conditioning.UNCONDITIONAL
        conditioning = None

    results = model.infer(
         data_idx=data_idx,
         num_grasps=args.num_grasps,
         visualize=args.visualize,
         condition_type=condition_type,
         conditioning=conditioning,
     )
    
    #print ("H: ",H, "Shape: ", H.shape)             #THE GRASPS ARE HERE
    #if args.visualize:
    #     results.show(line_settings={"point_size": 10})
    if Eval==True:
        from evaluation.grasp_quality_evaluation.evaluate_model import EvaluatePointConditionedGeneratedGrasps
        evaluator = EvaluatePointConditionedGeneratedGrasps(generator=model,args=args, n_grasps=args.num_grasps, obj_id=args.obj_id, obj_class=args.obj_class, n_envs=args.num_envs,
                                                        viewer=True, model_input=3,conditioning_type=condition_type)

        success_cases, edd_mean, edd_std = evaluator.generate_and_evaluate(success_eval=True, earth_moving_distance=True)

        pass

if __name__ == "__main__":
    main()
