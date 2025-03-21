from isaacgym import *
import os, sys
from tqdm import tqdm

import numpy as np
from DiffusionFields.se3dif.datasets import AcronymGraspsDirectory
from evaluation.grasp_sim.environments import IsaacGymWrapper
from evaluation.grasp_sim.environments.grip_eval_env import GraspingGymEnv

from DiffusionFields.se3dif.utils import to_numpy, to_torch
from evaluation.utils.geometry_utils import pq_to_H, H_2_Transform
import torch
import random

class GraspSuccessEvaluator():

    def __init__(self, obj_class, n_envs = 10, idxs=None, viewer=True, device='cpu', rotations=None, enable_rel_trafo=True,model_input=None):


        #self.model_input=1
        self.device = device
        self.obj_class = obj_class
        self.data_root_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),"data")
        self.grasps_directory = AcronymGraspsDirectory(data_type=self.obj_class)
        self.n_envs = n_envs
        self.rotations = rotations
        # This argument tells us if the grasp poses are relative w.r.t. the current object pose or not
        self.enable_rel_trafo = enable_rel_trafo

        ## Build Envs ##
        if idxs is None:
            idxs = [0]*n_envs                
            
        grasps = [self.grasps_directory.avail_obj[idx_i] for idx_i in idxs]         

        scales = [grasp.mesh_scale for grasp in grasps]
        obj_ids = [idx_i for idx_i in idxs]
        obj_types = [grasp.mesh_type for grasp in grasps]
        # Note: quaternion here already has convention, w,x,y,z
        if not(rotations is None):
            rotations_flat = [rotation for rotation in rotations]
        else:
            rotations_flat = [[0,0,0,1] for idx_i in idxs]

        env_args = self._get_args(obj_ids, obj_types, scales, rotations_flat)

        self.grasping_env = IsaacGymWrapper(env=GraspingGymEnv, env_args=env_args,
                                            z_convention=True, num_spaces=self.n_envs,
                                            viewer = viewer, device=self.device)

    
        self.success_cases = 0
        self.robust_cases = 0

        self.robust_fails=0
        self.fail_cases=0

        self.sing_values=[]
        self._grasps=[]

        self._force_closure=[]
        self.smallest_sing_value=[]
        
            
    def reset(self):
        self.success_cases = 0
        self.robust_cases = 0
        self.robust_fails=0
        self.fail_cases=0
        self.sing_values=[]
        self._grasps=[]

        self._force_closure=[]
        

    def _get_args(self, obj_ids, obj_types='Mug', scales=1.,rotations=None):
        args = []
        for obj_id, obj_type, scale, rotation in zip(obj_ids, obj_types, scales,rotations):
            obj_args = {
                'object_type': obj_type,
                'object_id': obj_id,
                'object_name': 'mug_01',
                'scale': scale,
                'obj_or': np.asarray(rotation)
            }
            arg = {'obj_args': obj_args}
            args.append(arg)
        return args

    def eval_set_of_grasps(self, H):
        n_grasps = H.shape[0]

        for i in tqdm(range(0, n_grasps, self.n_envs)):           
            print('iteration: {}'.format(i))

            batch_H = H[i:i+self.n_envs,...]
            self.eval_batch(batch_H)

        return self.success_cases,self.robust_cases,self.robust_fails,self.fail_cases, self.sing_values, self._grasps, self.grasping_env,self._force_closure

    def eval_batch(self, H):                                
        s = self.grasping_env.reset()                          
        for t in range(10):
            self.grasping_env.step()
        s = self.grasping_env.reset()
        for t in range(10):
            self.grasping_env.step()

        # 1. Set Evaluation Grasp
        self._grasps.append(H)
        H_obj = torch.zeros_like(H)
        for i, s_i in enumerate(s):
            H_obj[i,...] = pq_to_H(p=s_i['obj_pos'], q=s_i['obj_rot'])              

            if not(self.enable_rel_trafo):                                          # if relative transformation is disabled then replace the 3x3 of i-th matrix with the identitiy matrix
                H_obj[i, :3, :3] = torch.eye(3)                                     # due to this step, we only have the position of the object in H_obj now
        Hg = torch.einsum('bmn,bnk->bmk', H_obj, H)                                 # transformation of H using H_obj. as H_objs is only translation, it add the translation vector of npth the matrices 

        state_dicts = []
        for i in range(Hg.shape[0]):
            state_dict = {
                'grip_state': Hg[i,...]
            }
            state_dicts.append(state_dict)                                          

        s = self.grasping_env.reset(state_dicts)

        # 2. Grasp
        policy = GraspController(self.grasping_env, n_envs=self.n_envs)
        evaluate="pickup"
        T = 3500                                            

        for t in tqdm(range(T)):                      
            try:
                a = policy.control(s,evaluate)                                                   
                s = self.grasping_env.step(a)
                
                if t==700:
                    self._compute_success(s)
                    
                    Grasp_matrix=self.grasping_env.grasp_matrix()       
                    if Grasp_matrix.size==0:
                        S=np.array([-1.0,-1.0,-1.0,-1.0,-1.0,-1.0])
                        sigma=np.round(np.min(S),3)
                        self.sing_values.append(S)
                        self.smallest_sing_value.append(sigma)
                        
                    else:
                        a=np.isnan(Grasp_matrix).any()
                        b=np.any(np.all(Grasp_matrix ==0 ,axis=1))
                        if a or b:
                            S=np.array([-1.0,-1.0,-1.0,-1.0,-1.0,-1.0])
                            sigma=np.round(np.min(S),3)
                            self.sing_values.append(S)
                            self.smallest_sing_value.append(sigma)
                        else:
                            U,S,V=np.linalg.svd(Grasp_matrix)
                            sigma=np.round(np.min(S),3),
                            self.sing_values.append(S)
                            self.smallest_sing_value.append(sigma)
                    
                    
                    is_force_closure=self.grasping_env.force_closure(G=Grasp_matrix,S=S)
                    self._force_closure.append(is_force_closure)
                    
            except (IndexError):
                pass
                
            
        if policy.check_object_in_grasp(s)== True:  
            self.robust_cases +=1   

        
        del policy
        torch.cuda.empty_cache()



    def _compute_success(self, s):
        success_counter=0
        for i,si in enumerate(s):
            if si!="stop":
                hand_pos = si['hand_pos']
                obj_pos  = si['obj_pos']
                ## Check How close they are ##
                distance = (hand_pos - obj_pos).pow(2).sum(-1).pow(.5)              

                if distance <0.3:
                    self.success_cases +=1
                    success_counter+=1
                else:
                    self.fail_cases+=1
            else:
                pass


class GraspController():
    '''
     A controller to evaluate the grasping
    '''
    def __init__(self, env, hand_cntrl_type='position', finger_cntrl_type='torque', n_envs = 0):
        self.env = env

        ## Controller Type
        self.hand_cntrl_type = hand_cntrl_type
        self.finger_cntrl_type = finger_cntrl_type

        self.squeeze_force = .6
        self.hold_force = [self.squeeze_force]*n_envs

        self.r_finger_target = [0.]*n_envs
        self.l_finger_target = [0.]*n_envs
        self.grasp_count = [0]*n_envs
        self.evaluate="Pickup"

        ## State Machine States
        self.control_states = ['approach', 'grasp', 'lift','evaluation']
        self.grasp_states = ['squeeze', 'hold']
        self.evaluation_states= ['shake','rotate','rotate and translate','hold']

        self.state = ['grasp']*n_envs
        self.grasp_state = ['squeeze']*n_envs
        self.evaluation_state=['shake']*n_envs          

        ## Evaluation parameters
        k=0.4+ (0.4*random.random())
        #k=1
        print ("k = ", k)
        self.shake_amplitude = 0.05*k  # Increase amplitude for stronger shake
        self.shake_frequency = 10.0 *k
        self.target_rotation=self.targeted_rotation(mode="rotate")
        self.axis=self.target_axis()
        
        #self.translation_amplitude=k*0.05

        self.counter=0      #for 5 random rotation
        self.movement_mag= 0.05*k

        self.hold_counter=0 # for hold between the robustness check

    def targeted_rotation(self,mode=""):

        random_quat_vals=torch.randn(4)
        while torch.norm(random_quat_vals)<1e-6: 
                random_quat_vals=torch.randn(4)
        random_quat_vals/=torch.norm(random_quat_vals)
        target_rotation=gymapi.Quat(random_quat_vals[0],random_quat_vals[1],random_quat_vals[2],random_quat_vals[3])
        return target_rotation

    def target_axis(self):
        axis=torch.randn(3)
        while torch.norm(axis)<1e-6: 
            axis=torch.randn(3)
        axis/= torch.norm(axis)
        return axis

    def set_H_target(self, H):
        self.H_target = H
        self.T = H_2_Transform(H)
    
    def check_object_in_grasp(self, s):
        for si in s:
            hand_pos = si['hand_pos']
            obj_pos  = si['obj_pos']
            ## Check How close they are ##
            distance = (hand_pos - obj_pos).pow(2).sum(-1).pow(.5)
            if distance <0.3:
                return True
            else: False

    def check_object_in_grasp_singular(self, state):
        hand_pos = state['hand_pos']
        obj_pos  = state['obj_pos']
        ## Check How close they are ##
        distance = (hand_pos - obj_pos).pow(2).sum(-1).pow(.5)
        if distance <0.3:
            return True
        else: False


    def control(self, states, evaluate):
        self.evaluate=evaluate
        actions = []
#        if evaluate=="pickup":
        for idx, state in enumerate(states):
            if self.state[idx] =='approach':
                action = self._approach(state, idx)
            elif self.state[idx] == 'grasp':
                if self.grasp_state[idx] == 'squeeze':
                    action = self._squeeze(state, idx)
                elif self.grasp_state[idx] == 'hold':
                    action = self._hold(state, idx,mode="success")
            elif self.state[idx] == 'lift':
                action = self._lift(state, idx)
            elif self.state[idx] == 'evaluation':
                if self.evaluation_state[idx]=="shake":
                    action = self._shake(state,idx)
                    if not self.check_object_in_grasp_singular(state):
                        action="stop"

                elif self.evaluation_state[idx]=="jerk":
                    action = self._jerk(state,idx)
                    if not self.check_object_in_grasp_singular(state):
                        action="stop"

                elif self.evaluation_state[idx]=="rotate":
                    action = self._rotate(state,idx)
                    if not self.check_object_in_grasp_singular(state):
                        action="stop"

                elif self.evaluation_state[idx]=="rotate and translate":
                    action = self._rotate_and_translate(state,idx)
                    if not self.check_object_in_grasp_singular(state):
                        action="stop"

                elif self.grasp_state[idx] == 'hold':
                    action = self._hold(state, idx,mode="evaluate")
            actions.append(action)
        
        return actions

    def _approach(self, state, idx=0):
        hand_pose  = state[0]['hand_pos']
        hand_rot  = state[0]['hand_rot']

        target_pose = torch.Tensor([self.T.p.x, self.T.p.y, self.T.p.z])

        pose_error = hand_pose - target_pose
        des_pose = hand_pose - pose_error*0.1

        ## Desired Pos for left and right finger is 0.04
        r_error = state[0]['r_finger_pos'] - .04
        l_error = state[0]['l_finger_pos'] - .04

        K = 1000
        D = 20
        ## PD Control law for finger torque control
        des_finger_torque = torch.zeros(2)
        des_finger_torque[1:] += -K*r_error - D*state[0]['r_finger_vel']
        des_finger_torque[:1] += -K*l_error - D*state[0]['l_finger_vel']

        action = {'hand_control_type': 'position',
                  'des_hand_position': des_pose,
                  'grip_control_type': 'torque',
                  'des_grip_torque': des_finger_torque
                  }

        ## Set State Machine Transition
        error = pose_error.pow(2).sum(-1).pow(.5)
        if error<0.005:
            print('start grasp')
            self.state[idx] = 'grasp'
        return action

    def _squeeze(self, state, idx=0):
        ## Squeezing should achieve stable grasping / contact with the object of interest
        des_finger_torque = torch.ones(2)*-self.squeeze_force

        action = {'grip_control_type': 'torque',
                  'des_grip_torque': des_finger_torque}

        ## Set State Machine Transition after an empirical number of steps
        self.grasp_count[idx] +=1
        if self.grasp_count[idx]>300:
            self.grasp_count[idx] = 0
            self.grasp_state[idx] = 'hold'

        return action
    def _hold(self, state, idx=0,mode=""):
        
        if mode=="success":

            if self.grasp_count[idx] == 0:
                self.hold_force[idx] = self.squeeze_force           ### .6
            else:
                self.hold_force[idx] +=1.0                          ###1.6, 2.6.... so on and so forth,  
            self.grasp_count[idx] += 1

            ## Set torques
            des_finger_torque = torch.ones(2) * -self.hold_force[idx]       #### -1.6, -2.6 .. so on and so forth
            action = {'grip_control_type': 'torque',
                    'des_grip_torque': des_finger_torque}
            ## Set State Machine Transition after an empirical number of steps, this also corresponded
            ## to increasing the desired grasping force for 100 steps
            if self.grasp_count[idx] > 100:
                #print(self.hold_force[idx])

                self.grasp_count[idx] = 0.
                self.l_finger_target[idx] = state['l_finger_pos'].clone()
                self.r_finger_target[idx] = state['r_finger_pos'].clone()

                self.state[idx] = 'lift'

        elif mode=="evaluate":
            self.grasp_count[idx] += 1

            des_finger_torque = -self.hold_force[idx]       #### -1.6, -2.6 .. so on and so forth
            action = {'grip_control_type': 'torque',
                    'des_grip_torque': des_finger_torque}

            if self.hold_counter==0:                # between shake and rotate
                if self.grasp_count[idx] > 50:
                    #print(self.hold_force[idx])

                    self.grasp_count[idx] = 0.
                    self.l_finger_target[idx] = state['l_finger_pos'].clone()
                    self.r_finger_target[idx] = state['r_finger_pos'].clone()

                    self.evaluation_state[idx] = 'rotate'
            elif self.hold_counter==1:            #between rotate and random
                if self.grasp_count[idx] > 50:
                    #print(self.hold_force[idx])

                    self.grasp_count[idx] = 0.
                    self.l_finger_target[idx] = state['l_finger_pos'].clone()
                    self.r_finger_target[idx] = state['r_finger_pos'].clone()

                    self.evaluation_state[idx] = 'rotate and translate'
            elif self.hold_counter==2:
                if self.grasp_count[idx] > 10:
                    #print(self.hold_force[idx])

                    self.grasp_count[idx] = 0.
                    self.l_finger_target[idx] = state['l_finger_pos'].clone()
                    self.r_finger_target[idx] = state['r_finger_pos'].clone()

                    self.evaluation_state[idx] = 'rotate and translate'
            elif self.hold_counter==3:
                pass


        return action

    def _lift(self, state, idx=0):
        obj_pose = state['obj_pos']
        hand_pose  = state['hand_pos']
        #print('hand y: {}, obj y: {}'.format(hand_pose[1], obj_pose[1]))

        target_pose = torch.zeros_like(obj_pose)
        target_pose[2] = 2.
        target_pose[0] = 0.

        ## Set Desired Hand Pose
        pose_error = hand_pose - target_pose
        des_pose = hand_pose - pose_error*0.05

        ## Set Desired Grip Force
        des_finger_torque = torch.ones(2) * -self.hold_force[idx]

        r_error = state['r_finger_pos'] - self.r_finger_target[idx]
        l_error = state['l_finger_pos'] - self.l_finger_target[idx]

        K = 1000
        D = 20
        des_finger_torque[1:] += -K*r_error - D*state['r_finger_vel']
        des_finger_torque[:1] += -K*l_error - D*state['l_finger_vel']


        action = {'hand_control_type': 'position',
                  'des_hand_position': des_pose,
                  'grip_control_type': 'torque',
                  'des_grip_torque': des_finger_torque
                  }

        self.grasp_count[idx]+=1
        if self.grasp_count[idx] > 300:
            #print(self.hold_force[idx])
            self.grasp_count[idx] = 0
            self.state[idx] = 'evaluation'
        return action

    def _shake(self,state,idx=0):
        hand_pose = state['hand_pos']
        shake_amplitude = self.shake_amplitude  
        shake_frequency = self.shake_frequency   
        step = self.grasp_count[idx]
        # Total steps required for 5 oscillations per direction
        steps_per_oscillation = int((2 * np.pi) * shake_frequency)  # Steps for 1 oscillation
        total_oscillation_steps = 5* steps_per_oscillation  # 5 oscillations
        # Initialize shake phase tracking if not present
        if not hasattr(self, 'shake_phase'):
            self.shake_phase = [0] * len(self.state)  # 0 = x, 1 = y, 2 = z
        dx, dy, dz = 0.0, 0.0, 0.0                                              
        # Shake in the current phase direction
        if self.shake_phase[idx] == 0:  # X-direction
            dx = shake_amplitude * np.sin(step / shake_frequency)                          # Sine for controlled and periodic perturbation
            #dx =shake_amplitude *np.exp(-0.01*step)* np.sin(step/shake_frequency)          # Exponentially Damped Sine for real worl decay. Reduces shaking over time but is not good for sustaining
            #dx=shake_amplitude *2* np.arcsin(np.sin(step/shake_frequency)) /np.pi           # Triangle Wave which is periodic but sharper for jerky but smooth motions
        elif self.shake_phase[idx] == 1:  # Y-direction
            dy = shake_amplitude * np.sin(step / shake_frequency)
            #dy =shake_amplitude *np.exp(-0.01*step)* np.sin(step/shake_frequency)
            #dy=shake_amplitude *2* np.arcsin(np.sin(step/shake_frequency)) /np.pi
        elif self.shake_phase[idx] == 2:  # Z-direction
            dz = shake_amplitude * np.sin(step / shake_frequency)
            #dz =shake_amplitude *np.exp(-0.01*step)* np.sin(step/shake_frequency)
            #dz=shake_amplitude *2* np.arcsin(np.sin(step/shake_frequency)) /np.pi
        # Apply the shaking motion by modifying hand position
        des_pose = hand_pose + torch.tensor([dx, dy, dz])
        action = {'hand_control_type': 'position',
                  'des_hand_position': des_pose,
                  }
        
        if step >= total_oscillation_steps:
            self.grasp_count[idx] = 0  # Reset step count for next phase
            self.shake_phase[idx] += 1  # Move to the next phase
        if self.shake_phase[idx] > 2:
            self.grasp_count[idx] = 0.
            self.evaluation_state[idx] = 'hold'
        self.grasp_count[idx] += 1
        return action

    

    def _rotate(self,state,idx=0):
        if self.hold_counter==0:
            self.hold_counter=1

        current_rot=state['hand_rot']
        delta_rot=self.target_rotation
        des_pose=torch.tensor([delta_rot.x,delta_rot.y,delta_rot.z,delta_rot.w])

        action = {'hand_control_type': 'orientation',
                'des_hand_position': des_pose,
                }

        self.grasp_count[idx]+=1
        if self.grasp_count[idx] > 150:
            self.counter+=1
            self.grasp_count[idx]=0
            self.target_rotation=self.targeted_rotation(mode="rotate")
            if self.counter>4:
                self.counter=0
                self.evaluation_state[idx] = 'hold'
        return action

        

    def _rotate_and_translate(self,state,idx=0):
        if self.hold_counter==0:
            self.hold_counter=2
        if self.hold_counter==1 and self.counter<5:            #if need be to move into another evaluation phase
            self.hold_counter=2

        if self.hold_counter==2:
            self.hold_counter=1

        current_rot=state['hand_rot']
        delta_rot=self.target_rotation
        hand_pos=state['hand_pos']
        des_hand_pose=hand_pos+ self.axis*self.movement_mag
        des_pose=torch.tensor([des_hand_pose[0],des_hand_pose[1],des_hand_pose[2],delta_rot.x,delta_rot.y,delta_rot.z,delta_rot.w])
        action = {'hand_control_type': 'random',
                'des_hand_position': des_pose,
                }

        self.grasp_count[idx]+=1
        
        if self.grasp_count[idx] > 150:
            self.evaluation_state[idx]= 'hold'
            self.counter+=1
            self.grasp_count[idx]=0
            self.target_rotation=self.targeted_rotation(mode="random")
            self.axis=self.target_axis()
            if self.counter>4:
                self.evaluation_state[idx]= 'hold'
                if self.hold_counter==1 and self.counter==5:            
                    self.hold_counter=3                
                    self.counter=0

        return action