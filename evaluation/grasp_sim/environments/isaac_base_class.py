from isaacgym import gymapi, gymtorch
from isaacgym import gymutil
import math
import time
import os, os.path as osp
import copy
from scipy.spatial.transform import Rotation as R
import numpy as np


import torch
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
from itertools import combinations
from scipy.linalg import svd

PHYSICS = 'PHYSX'


class IsaacGymWrapper():

    def __init__(self, env, viewer=True, physics=PHYSICS, freq = 250, device = 'cuda:0',
                 num_spaces = 1, env_args=None, z_convention=False):
        #self.reset
        self.franka_urdf = None 

        ## Args
        self.sim_device_type, self.compute_device_id = gymutil.parse_device_str(device)
        self.device = device
        self.physics = physics
        self.freq = freq
        self.num_spaces = num_spaces
        self._set_transforms()
        self.z_convention = z_convention

        self.visualize = viewer



        self.gym = gymapi.acquire_gym()
        #if IsaacGymWrapper.sim==None:
        
        self.sim, self.sim_params = self._create_sim()

        #    IsaacGymWrapper.sim=self.sim
        #else:
        #    pass

        ## Create Visualizer
        #if IsaacGymWrapper.viewer==None:
        if (self.visualize):
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            if self.viewer is None:
                print("*** Failed to create viewer")
                quit()
        else:
            self.viewer = None
            
        #else:
        #    pass

        ## Create Environment
        self._create_envs(env, env_args)

        ## Update camera pose
        if self.visualize:
            self._reset_camera(env_args)

    def _create_sim(self):
        """Set sim parameters and create a Sim object."""
        # Set simulation parameters

        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.dt = 1.0 / self.freq
        sim_params.gravity = gymapi.Vec3(0, -9.81, 0)
        sim_params.substeps = 1

        # Set stress visualization parameters
        sim_params.stress_visualization = True
        sim_params.stress_visualization_min = 1.0e2
        sim_params.stress_visualization_max = 1e5

        if self.z_convention:
            sim_params.up_axis = gymapi.UP_AXIS_Z
            sim_params.gravity = gymapi.Vec3(0., 0., -9.81)

        if self.physics == 'FLEX':
            sim_type = gymapi.SIM_FLEX
            print('using flex engine...')

            # Set FleX-specific parameters
            sim_params.flex.solver_type = 5
            sim_params.flex.num_outer_iterations = 10
            sim_params.flex.num_inner_iterations = 200
            sim_params.flex.relaxation = 0.75
            sim_params.flex.warm_start = 0.8

            sim_params.flex.deterministic_mode = True

            # Set contact parameters
            sim_params.flex.shape_collision_distance = 5e-4
            sim_params.flex.contact_regularization = 1.0e-6
            sim_params.flex.shape_collision_margin = 1.0e-4
            sim_params.flex.dynamic_friction = 0.7
        else:
            sim_type = gymapi.SIM_PHYSX
            print("using physx engine")
            sim_params.physx.solver_type = 1
            sim_params.physx.num_position_iterations = 25
            sim_params.physx.num_velocity_iterations = 1
            sim_params.physx.num_threads = 2
            sim_params.physx.use_gpu = True

            sim_params.physx.rest_offset = 0.0
            sim_params.physx.contact_offset = 0.001
            sim_params.physx.friction_offset_threshold = 0.001
            sim_params.physx.friction_correlation_distance = 0.0005

        # Create Sim object
        gpu_physics = self.compute_device_id
        if self.visualize:
            gpu_render = 0
        else:
            gpu_render = -1

        return self.gym.create_sim(gpu_physics, gpu_render, sim_type,
                                   sim_params), sim_params

    def _create_envs(self, env, env_args):
        # Add ground plane
        plane_params = gymapi.PlaneParams()
        if self.z_convention:
            plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

        # Set up the env grid - only 1 object for now
        num_envs = self.num_spaces
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        # Some common handles for later use
        self.envs_gym = []
        self.envs = []
        print("Creating %d environments" % num_envs)
        num_per_row = int(math.sqrt(num_envs))

        for i in range(num_envs):
            if isinstance(env_args, list):
                env_arg = env_args[i]
            else:
                env_arg = env_args

            # create env
            env_i = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs_gym.append(env_i)
            # now initialize the respective env:
            self.envs.append(env(self.gym, self.sim, env_i, self, i, env_arg))

    def _set_transforms(self):
        """Define transforms to convert between Trimesh and Isaac Gym conventions."""
        self.from_trimesh_transform = gymapi.Transform()
        self.from_trimesh_transform.r = gymapi.Quat(0, 0.7071068, 0,
                                                    0.7071068)
        self.neg_rot_x_transform = gymapi.Transform()
        self.neg_rot_x = gymapi.Quat(0.7071068, 0, 0, -0.7071068)
        self.neg_rot_x_transform.r = self.neg_rot_x

    def _reset_camera(self, args):
        if self.z_convention is False:
            # Point camera at environments
            cam_pos = gymapi.Vec3(0.0, 1.0, 0.6)
            cam_target = gymapi.Vec3(0.0, 0.8, 0.2)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        else:
            # Point camera at environments
            if args is not None:
                if 'cam_pose' in args:
                    cam = args['cam_pose']
                    cam_pos = gymapi.Vec3(cam[0], cam[1], cam[2])
                    cam_target = gymapi.Vec3(cam[3], cam[4], cam[5])
                else:
                    cam_pos = gymapi.Vec3(0.0, 0.9, 1.3)
                    cam_target = gymapi.Vec3(0.0, 0.0, .7)
            else:
                cam_pos = gymapi.Vec3(0.0, 0.9, 1.3)
                cam_target = gymapi.Vec3(0.0, 0.0, .7)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def get_franka_rpy(self, trimesh_grasp_quat):
        """Return RPY angles for Panda joints based on the grasp pose in the Z-up convention."""
        neg_rot_x = gymapi.Quat(0.7071068, 0, 0, -0.7071068)
        rot_z = gymapi.Quat(0, 0, 0.7071068, 0.7071068)
        desired_transform = neg_rot_x * trimesh_grasp_quat * rot_z
        r = R.from_quat([
            desired_transform.x, desired_transform.y, desired_transform.z,
            desired_transform.w
        ])
        return desired_transform, r.as_euler('ZYX')

    def reset(self, state=None):
        '''
        The reset function receives a list of dictionaries with the desired reset state for the different elements
        in the environment.
        '''
        for idx, env_i in enumerate(self.envs):
            if state is not None:
                env_i.reset(state[idx])
            else:
                env_i.reset()

        return self._evolve_step()

    def reset_robot(self, state=None, ensure_gripper_reset=False):
        '''
        The reset function receives a list of dictionaries with the desired reset state for the different elements
        in the environment. This function only resets the robot
        '''
        # if gripper reset should be ensured, we require two timesteps:
        if (ensure_gripper_reset):
            for idx, env_i in enumerate(self.envs):
                if state is not None:
                    env_i.reset_robot(state[idx],zero_grip_torque=True)
                else:
                    env_i.reset_robot(zero_grip_torque=True)
            self._evolve_step()


        for idx, env_i in enumerate(self.envs):
            if state is not None:
                env_i.reset_robot(state[idx])
            else:
                env_i.reset_robot()

        return self._evolve_step()

    def reset_obj(self, state=None):
        '''
        The reset function receives a list of dictionaries with the desired reset state for the different elements
        in the environment. This function only resets the robot
        '''
        for idx, env_i in enumerate(self.envs):
            if state is not None:
                env_i.reset_obj(state[idx])
            else:
                env_i.reset_obj()

        return self._evolve_step()

    def get_state(self):

        # refresh tensors
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # Rigid body state tensor
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        rb_states = gymtorch.wrap_tensor(_rb_states)
        rb_states = rb_states.view(self.num_spaces, -1, rb_states.shape[-1])

        # DOF state tensor
        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        dof_states = gymtorch.wrap_tensor(_dof_states)
        n_dofs = self.envs[0].n_dofs
        dof_vel = dof_states[:, 1].view(self.num_spaces, n_dofs, 1)
        dof_pos = dof_states[:, 0].view(self.num_spaces, n_dofs, 1)

        s = []
        for idx, env_i in enumerate(self.envs):
            s.append(env_i.get_state([rb_states[idx,...], dof_pos[idx, ...], dof_vel[idx, ...]]))

        return s

    def step(self, action=None):

        if action is not None:
            for idx, env_i in enumerate(self.envs):
                if action[idx]!="stop":    
                    env_i.step(action[idx])
                else:
                    #print("bad grasp")
                    pass

        return self._evolve_step()

    def _evolve_step(self):
        # get the sim time
        t = self.gym.get_sim_time(self.sim)

        # Step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        # refresh tensors
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # Step rendering
        self.gym.step_graphics(self.sim)

        if self.visualize:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)

        return self.get_state()

    def kill(self):
        self.gym.destroy_viewer(self.viewer)
        for env in self.envs_gym:
            self.gym.destroy_env(env)
        self.gym.destroy_sim(self.sim)

    def grasp_matrix(self):
        ### Rigid Body Properties ###
        rigid_contacts=self.gym.get_env_rigid_contacts(self.envs_gym[0])    # [('env0', '<i4'), ('env1', '<i4'), ('body0', '<i4'), ('body1', '<i4'),  ('localPos0', [...]), ('localPos1', [...]), ('minDist', '<f4'), ('initialOverlap', '<f4'), ('normal', [...]), ('offset0', [...]), ('offset1', [...]), ('lambda', '<f4'), ('lambdaFriction', [...]), friction, torsionalfriction, rolling friction]

        ### GET COM OF THE OBEJCT ####
        properties=self.gym.get_actor_rigid_body_properties(self.envs_gym[0],self.envs[0].shape_obj.handle)
        COM=[body.com for body in properties]
        COM=np.array([COM[0].x,COM[0].y,COM[0].z],dtype=np.float64)

        ### Grasp Matrix ###
        G=self.make_grasp_matrix(rigid_contacts,COM)
        #print("Grasp_matrix: ", G)
        return G

    def make_grasp_matrix(self,contact_data, COM=None):
        num_contacts=len(contact_data)
        G=np.zeros((6,3*num_contacts))
        for i, contact in enumerate(contact_data):

            p0=np.array(contact[4])
            p0=np.array([p0['x'],p0['y'],p0['z']],dtype=np.float64)
            p1=np.array(contact[5])
            p1=np.array([p1['x'],p1['y'],p1['z']],dtype=np.float64)
            normal=np.array(contact[8])
            normal=np.array([normal['x'],normal['y'],normal['z']],dtype=np.float64)
            lambda_force=contact[11]
            
            p=(p0 + p1)/2
            p=p - COM

#            G[:3, 3*i: 3*i +3]=np.eye(3)               incase no normal force or anything like that. we have normal and lambda so we will use them.
            force=lambda_force*normal
            G[:3, 3*i: 3*i +3]=np.diag(force)
            r_cross =np.array([[0,-p[2],p[1]],[p[2],0,-p[1]],[-p[1],p[0],0]])
            G[3:,3*i:3*i +3]=r_cross

        return G

    def visualise_contact(self,gym,sim,viewer,envs,contact_data,scale=0.1):
        for contact in contact_data:
            env_id=contact['env0']
            if env_id==-1:
                continue
            else:
                env=envs[env_id]
                #contact_point=np.array([contact[4]['x'],contact[4]['y'],contact[4]['z']],dtype=np.float64)
                contact_point=contact[5]                    #outside contact[4] is outside points       and contact[5] is inside points
                normal=np.array([contact[8]['x'],contact[8]['y'],contact[8]['z']],dtype=np.float64)
                #normal=contact[8]
                force_mag=contact[11]

                rigid_body_handle= contact[2]

                if rigid_body_handle !=-1:
                    transform=gym.get_rigid_transform(env,rigid_body_handle)
                    contact_point_world=transform.p + transform.r.rotate(contact_point)
                else:
                    contact_point_world= contact_point

                force_x=gymapi.Vec3(force_mag*normal[0]*scale,0.0,0.0)
                force_y=gymapi.Vec3(0.0,force_mag*normal[1]*scale,0.0)
                force_z=gymapi.Vec3(0.0,0.0,force_mag*normal[2]*scale)

                force_end_x=contact_point_world + force_x
                force_end_y=contact_point_world + force_y
                force_end_z=contact_point_world + force_z

                color_x=gymapi.Vec3(1.0,0.0,0.0)    #Red
                color_y=gymapi.Vec3(0.0,1.0,0.0)    #Green
                color_z=gymapi.Vec3(0.0,0.0,1.0)    #Blue
                color_contact_0=gymapi.Vec3(1.0,1.0,0.0)    #Yellow for contact point 0

                gymutil.draw_line(contact_point_world,force_end_x,color_x,gym,viewer,env)
                gymutil.draw_line(contact_point_world,force_end_y,color_y,gym,viewer,env)
                gymutil.draw_line(contact_point_world,force_end_z,color_z,gym,viewer,env)
                
                contact_point_end=contact_point_world+gymapi.Vec3(0.0001,0.0001,0.0001)   # a little offset
                gymutil.draw_line(contact_point_world,contact_point_end,color_contact_0,gym,viewer,env)
                #force_vec=force_mag * normal*scale
                #force_vec=gymapi.Vec3(force_vec[0],force_vec[1],force_vec[2])
                #force_end=contact_point_world+ force_vec

                #color=gymapi.Vec3(1.0,0.0,0.0)
                #gymutil.draw_line(contact_point_world,force_end,color,gym,viewer,env)
        input ("Press Enter to continue....")
    
    def force_closure(self,G,S):
        if G.size!=0:
            threshold= 0.1
            #Condition 1: Rank check for G
            is_full_rank_1=np.count_nonzero(S)==6
            p,m=G.shape
            is_full_rank_2=np.linalg.matrix_rank(G)
            if is_full_rank_2<p:
                is_full_rank_2=False
            is_full_rank=is_full_rank_2 and is_full_rank_1

            #Condition 2: Threshold of smallest singular value
            is_stable=np.round(np.min(S),3) >= threshold

            ###Condition 3: Convex Hull
            #Checking if neighbourhood of the origin AND the original is in the convex hull
            try:
                epsilon=1e-6
                perturbations=np.random.uniform(-0.001,0.001,size=(10,6))
                hull=ConvexHull(G.T)
                zero_in_hull=np.all(hull.equations[:, :-1] @ np.zeros(6) + hull.equations[: , -1]<=epsilon)
                neighbourhood_in_all= all(np.all(hull.equations[:, :-1] @ p + hull.equations[: , -1]<=epsilon) for p in perturbations)
                _convex_hull=zero_in_hull and neighbourhood_in_all
            except:
                _convex_hull=False
            
            #Condition 4: Checking random wrenches generated
            _random_wrench=True
            num_contacts=G.shape[1]
            for i in range(10):
                #w=np.random.random(6)
                w=np.random.uniform(-1,1,6)
                c=np.zeros(num_contacts)
                A_eq=G
                b_eq=w
                bounds=[(0,None)] * num_contacts
                result=linprog(c,A_eq=A_eq,b_eq=b_eq,bounds=bounds, method='highs')
                if result.status !=0:
                    _random_wrench=False
            
            #if is_full_rank and is_stable and _random_wrench and _convex_hull:
            return {"is_stable": bool(is_stable), "is_full_rank": bool(is_full_rank), "Convex_hull": bool(_convex_hull), "random wrench": bool(_random_wrench)}
            #else:
                #return -1
        else:
           return {"is_stable": "No Grasp Matrix", "is_full_rank": "No Grasp Matrix", "Convex_hull": "No Grasp Matrix", "random wrench": "No Grasp Matrix"} 
        