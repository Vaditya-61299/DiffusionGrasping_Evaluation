from evaluation.grasp_quality_evaluation.grasps_sucess import GraspSuccessEvaluator

from scipy.spatial.transform import Rotation as R

from DiffusionFields.se3dif.utils import to_numpy, to_torch
from DiffusionFields.se3dif.datasets.acronym_dataset import AcronymGraspsDirectory
from graspLDM.grasp_ldm.dataset.acronym.acronym_pointclouds import AcronymBaseDataset, AcronymShapenetPointclouds
import numpy as np
import torch
import json



class EvaluatePointConditionedGeneratedGrasps():

    def __init__(self, generator, n_grasps = 500, batch=100, obj_id= 0, obj_class = 'Mug', n_envs=10, net_scale=8.,
                 viewer=True, args=None, center_P=True,model_input=None,conditioning_type=None):

        ## Set args
        self.center_P = center_P
        self.n_grasps = n_grasps
        self.n_envs = n_envs
        self.net_scale = net_scale
        self.obj_id = obj_id
        self.obj_class = obj_class
        self.batch = batch
        self.viewer = viewer
        self.args = self._set_args(args)
        self.model_input=model_input
#        self.args=args
        if self.model_input==3:
            self.conditioning_type=conditioning_type

        ## Load generator model
        self.generator = generator      # the grasps if graspLDM

    def _set_args(self, args):
        if args is None:
            args = {
                'generation_batch':self.batch,
                'empirical_dist_episodes':1000,
            }
        return args

    def generate_and_evaluate(self, success_eval=True, earth_moving_distance=True):
        H = self.generate_grasps()
        if success_eval:
            self.success_cases,self.robust_cases,self.robust_fails,self.fail_cases,self._sing_values,self._grasps,self.grasping_env,self._force_closure = self.evaluate_grasps_success(H)
            success_rate = self.success_cases/H.shape[0]  
        else:
            success_rate = 0.
        if earth_moving_distance:
            print('EMD Distance: Dataset-Dataset')
            self.EMD_DD=self.measure_empirircal_dist_distance()
            print('EMD Distance: Samples-Dataset')
            self.EMD_SD = self.measure_empirircal_dist_distance(H)
        else:
            edd_mean = 0.
            edd_std = 0.
        self.results()
        self.reset()
        return success_rate, self.EMD_SD[0],self.EMD_SD[1]

    def pointcloud_conditioning(self,results=None):
        if self.model_input!=3:
            acronym_grasps = AcronymGraspsDirectory(data_type=self.obj_class)
            mesh = acronym_grasps.avail_obj[self.obj_id].load_mesh()
            P = mesh.sample(1024)
            P = to_torch(P, self.generator.device)
            rot = to_torch(R.from_quat(self.q).as_matrix(), self.generator.device)
            P = torch.einsum('mn,bn->bm', rot, P)
            self.P = P.clone()


            P *=self.net_scale
            if self.center_P:
                self.P_mean = torch.mean(P, 0)
                P += -self.P_mean

            self.generator.model.set_latent(P[None,...], batch=self.generator.batch)
        else:
            print ("GRASPLDM NOT IMPLEMENTED YET")
            pass
    def generate_grasps(self, n_grasps=None):
        
        if n_grasps == None:
            n_grasps = self.n_grasps

        #q = np.random.randn(4)
        q = np.array([0., 0., 0., 1.])
        self.q = q/np.linalg.norm(q)
        if self.model_input!=3:
                ## Set a Random Rotations ##
            '''q = np.random.randn(4)
            #q = np.array([0., 0., 0., 1.])
            self.q = q/np.linalg.norm(q)'''
        ## Set SE3 Langevin Dynamics for generating Grasps
            self.pointcloud_conditioning()

        ## Generate Grasps in batch
            H = torch.zeros(0,4,4).to(self.generator.device)
            batch = self.generator.batch
            for i in range(0, self.n_grasps, batch):
                print('Generating of {} to {} samples'.format(i, i+batch))
                if self.model_input==1:
                    H_episode = self.generator.sample(batch=batch)
                elif self.model_input==2:
                    H_episode,t = self.generator.sample(batch=batch)
                ## Shift to CoM of the object
                if self.center_P:
                    H_episode[:, :3, -1] = H_episode[:, :3, -1] + self.P_mean
                ## Rescale
                H_episode[:, :3, -1] = H_episode[:, :3, -1]/self.net_scale

                H = torch.cat((H, H_episode), 0)


            return H
        else:
            print ("GRASPLDM NOT IMPLEMENTED YET")
            pass
    def evaluate_grasps_success(self, H):

        grasp_evaluator = GraspSuccessEvaluator(n_envs=self.n_envs, idxs=[self.obj_id] * self.n_envs, obj_class=self.obj_class,
                                                rotations=[self.q]*self.n_envs, viewer=self.viewer, enable_rel_trafo=False)

        return grasp_evaluator.eval_set_of_grasps(H)

    def measure_empirircal_dist_distance(self, H_sample=None):

        from scipy.optimize import linear_sum_assignment
        from pytorch3d.transforms.so3 import so3_rotation_angle

        ## Load Acronym Dataset and Sample n_grasps ##
        grasps_directory = AcronymGraspsDirectory(data_type=self.obj_class)
        grasps = grasps_directory.avail_obj[self.obj_id]
        Hgrasps = grasps.good_grasps

        rot = R.from_quat(self.q).as_matrix()
        H_map = np.eye(4)
        H_map[:3,:3] = rot
        Hgrasps = np.einsum('mn,bnd->bmd', H_map, Hgrasps)

        # Set Samples
        if H_sample is None:
            try:
                idx = np.random.randint(0, Hgrasps.shape[0], self.n_grasps)
            except:
                mean=0
                std=0
                print("No good grasps available")
                return mean,std
            H_sample = torch.Tensor(Hgrasps[idx, ...])
        p_sample = H_sample[:, :3, -1]
        R_sample = H_sample[:, :3, :3]


        divergence = np.zeros(0)
        for k in range(self.args['empirical_dist_episodes']):
            ## Sample Candidates ##
            idx = np.random.randint(0, Hgrasps.shape[0], self.n_grasps)
            H_eval = torch.Tensor(Hgrasps[idx, ...]).to(H_sample)

            p_eval = H_eval[:, :3, -1]
            R_eval = H_eval[:, :3, :3]

            xyz_dist = (p_eval[None,...] - p_sample[:,None,...]).pow(2).sum(-1).pow(.5)
            R12 = torch.einsum('bmn,knd->bkmd',R_eval.transpose(-1,-2) , R_sample)

            R12_ = R12.reshape(-1, 3, 3)
            R_dist_  = (1.-so3_rotation_angle(R12_, cos_angle=True))
            R_dist = R_dist_.reshape(R12.shape[0], R12.shape[1])

            distance = xyz_dist + R_dist

            distance = to_numpy(distance)
            row_ind, col_ind = linear_sum_assignment(distance)
            min_distance = distance[row_ind, col_ind].mean()
            divergence = np.concatenate((divergence, np.array([min_distance])),0)

        mean = np.mean(divergence)
        std = np.std(divergence)
        print('Wasserstein Distance Mean: {}, Variance: {}'.format(mean, std))
        return mean, std

    def results(self):
        _result={}
        _result[f"Object Class: {self.obj_class}"]={
            "Object ID": self.obj_id,
            "Success cases": self.success_cases,
            "Robust cases": self.robust_cases,
            "Fail Cases": self.fail_cases,
            "EMD-Dataset-Dataset":self.EMD_DD,
            "EMD-Sample-Dataset":self.EMD_SD,
            "Grasps":[]
        }

        for i in range(len(self._grasps)):


            grasp={
                "Grasp Index": i+1,
                "Grasp matrix": self._grasps[i].tolist(),
                "Singular Values of Grasp Matrix":self._sing_values[i].tolist(),
                "Smallest Singular Value of Grasp matrix":np.round(np.min(self._sing_values[i]),3),
                "Force Closure":self._force_closure[i]              #(is_stable   (small sing val crosses the threshold)      ,is_full_rank(==6)    ,_convex_hull(),  _random_wrench(able to resist any rando random wrench))
            }
            _result[f"Object Class: {self.obj_class}"]["Grasps"].append(grasp)
        with open("Results.json","a") as f:
            json.dump(_result,f,indent=4,default=str)
    
    def reset(self):
        self.success_cases=0
        self.fail_cases=0
        self.robust_cases=0
        self.robust_fails=0
        self._sing_values=[]
        self._grasps=[]
        self.grasping_env.gym.destroy_sim(self.grasping_env.sim)
        if self.grasping_env.viewer!=None:
            self.grasping_env.gym.destroy_viewer(self.grasping_env.viewer)
        
        