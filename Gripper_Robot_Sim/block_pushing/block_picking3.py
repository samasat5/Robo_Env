import enum
import collections
from gym import spaces
from gym.envs import registration
from typing import Dict, List, Optional, Tuple, Union
from block_pushing.utils.pose3d_gripper import Pose3d_gripper
from block_pushing.utils.pose3d import Pose3d
from block_pushing.utils.franka_panda_sim_robot import GripperArmSimRobot
from block_pushing.utils import utils_pybullet, franka_panda_sim_robot
from block_pushing.utils.utils_pybullet import ObjState, XarmState
import pybullet_utils.bullet_client as bullet_client
from scipy.spatial import transform
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pybullet
import time
import math
import gym
import pdb
import os


X_MIN_REAL = 0.15
X_MAX_REAL = 0.6
Y_MIN_REAL = -0.3048
Y_MAX_REAL = 0.3048
DEFAULT_CAMERA_POSE = (1.1, 0, 0.75)
DEFAULT_CAMERA_ORIENTATION = (np.pi / 4, np.pi, -np.pi / 2)
IMAGE_WIDTH = 320
EFFECTOR_HEIGHT = 0.06
IMAGE_HEIGHT = 240
CAMERA_INTRINSICS = (0.803 * IMAGE_WIDTH, 0,IMAGE_WIDTH / 2.0,0,0.803 * IMAGE_WIDTH,IMAGE_HEIGHT / 2.0,0,0,1,)
WORKSPACE_BOUNDS = np.array(((0.15, -0.5), (0.7, 0.5)))
WORKSPACE_BOUNDS_REAL = np.array(((X_MIN_REAL, Y_MIN_REAL), (X_MAX_REAL, Y_MAX_REAL)))
WORKSPACE_URDF_PATH_REAL = "third_party/py/ibc/environments/assets/workspace_real.urdf"
IMAGE_WIDTH_REAL = 320
IMAGE_HEIGHT_REAL = 180
CAMERA_POSE_REAL = (0.75, 0, 0.5)
CAMERA_ORIENTATION_REAL = (np.pi / 5, np.pi, -np.pi / 2)
CAMERA_INTRINSICS_REAL = (0.803 * IMAGE_WIDTH_REAL,0,IMAGE_WIDTH_REAL / 2.0,0,0.803 * IMAGE_WIDTH_REAL,IMAGE_HEIGHT_REAL / 2.0,0,0,1,)
BLOCK_URDF_PATH = "third_party/py/envs/assets/block2.urdf"
PLANE_URDF_PATH = "third_party/bullet/examples/pybullet/gym/pybullet_data/" "plane.urdf"
WORKSPACE_URDF_PATH = "third_party/py/envs/assets/workspace.urdf"
ZONE_URDF_PATH = "third_party/py/envs/assets/zone.urdf"
INSERT_URDF_PATH = "third_party/py/envs/assets/insert.urdf"
INITIAL_JOINT_POSITIONS = np.array(
    [
        0.0, 
        -0.5235987755982988, 
        0.0, 
        -1.0471975511965976, 
        0.0, 
        1.5707963267948966, 
        0.0, 
        0.0, 
        0.0
    ])

class BlockPick(gym.Env):
    def __init__(
    self,
    control_frequency=10.0,
    image_size=np.array([320, 240]),
    shared_memory=False, 
    seed=44,
    goal_dist_tolerance=0.01,
    effector_height=None,
    visuals_mode="default",
    # abs_action=False
):
      
        # Mimic RealSense D415 camera parameters.
        if visuals_mode != "default" and visuals_mode != "real":
            raise ValueError("visuals_mode must be `real` or `default`.")
        self._visuals_mode = visuals_mode
        if visuals_mode == "default":
            self._camera_pose = DEFAULT_CAMERA_POSE
            self._camera_orientation = DEFAULT_CAMERA_ORIENTATION
            self.workspace_bounds = WORKSPACE_BOUNDS
            self._camera_instrinsics = CAMERA_INTRINSICS
            self._workspace_urdf_path = WORKSPACE_URDF_PATH
        else:
            self._camera_pose = CAMERA_POSE_REAL
            self._camera_orientation = CAMERA_ORIENTATION_REAL
            self.workspace_bounds = WORKSPACE_BOUNDS_REAL
            self._camera_instrinsics = CAMERA_INTRINSICS_REAL
            self._workspace_urdf_path = WORKSPACE_URDF_PATH_REAL
        
        self._connection_mode = pybullet.GUI
        self._pybullet_client = bullet_client.BulletClient(connection_mode=self._connection_mode)
        if shared_memory:
            self._connection_mode = pybullet.SHARED_MEMORY
        self._setup_the_scene()
        self._image_size = image_size
        self.image_size = np.array([320, 240])
        self.image_dir="trajectory_images"
        self.save_state()
        self.effector_height = effector_height or EFFECTOR_HEIGHT
        self.offset = np.array([0.03, 0, 0])  # assume fingers are 6cm apart
        self._target_effector_pose = None
        self._target_pose = None
        self._control_frequency = control_frequency
        assert isinstance(self._pybullet_client, bullet_client.BulletClient)
        self.workspace_center_x = 0.4
        self._is_grasped = False
        self._rng = np.random.RandomState(seed=seed)
        self.block_translation = None
        self.rendered_img = None
        self.goal_dist_tolerance =  goal_dist_tolerance
    
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(3,))  
        self.observation_space = self._create_observation_space(image_size)
        # pybullet.startStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4, "simulation_video.mp4")
        # self.reset()


    @property
    def pybullet_client(self):
        return self._pybullet_client
    @property
    def robot(self):
        return self._robot
    @property
    def workspace_uid(self):
        return self._workspace_uid
    @property
    def target_effector_pose(self):
        return self._target_effector_pose
    @property
    def target_pose(self):
        return self._target_pose
    @property
    def control_frequency(self):
        return self._control_frequency
    @property
    def connection_mode(self):
        return self._connection_mode
    
    def step_Simulation_func(self, nsteps=100,timesleep=0.01):
        for _ in range(nsteps):
            self._pybullet_client.stepSimulation()
            time.sleep(timesleep)

        
    
    
    def _setup_the_scene(self):
        
        # bullet_client.BulletClient(connection_mode=self._connection_mode)
        self._pybullet_client.resetSimulation()
        self._pybullet_client.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
        self._pybullet_client.setPhysicsEngineParameter(enableFileCaching=0)
        self._pybullet_client.setGravity(0, 0, -9.8)

        utils_pybullet.load_urdf(
            self._pybullet_client, PLANE_URDF_PATH, basePosition=[0, 0, -0.001])
        
        self._workspace_uid = utils_pybullet.load_urdf(
            self._pybullet_client,
            self._workspace_urdf_path,
            basePosition=[0.35, 0, 0.0],)
        self._robot = franka_panda_sim_robot.GripperArmSimRobot(  
            self._pybullet_client,
            initial_joint_positions=INITIAL_JOINT_POSITIONS, 
            color="white" if self._visuals_mode == "real" else "default",)
        print("INITIAL DONE")
        self._target_id = utils_pybullet.load_urdf(self._pybullet_client, ZONE_URDF_PATH, useFixedBase=True)
        self._block_id = utils_pybullet.load_urdf(self._pybullet_client, BLOCK_URDF_PATH, useFixedBase=False)
        
        self._pybullet_client.createConstraint(   
            parentBodyUniqueId=self._workspace_uid,
            parentLinkIndex=-1,                # -1 means the base link of the parent
            childBodyUniqueId=self._target_id ,
            childLinkIndex=-1,                 # -1 means the base link of the child
            jointType=self._pybullet_client.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0.02],   # Position of zone relative to workspace
            childFramePosition=[0, 0, 0],      # Position of zone relative to its own origin
        )
        self._pybullet_client.resetDebugVisualizerCamera(
            cameraDistance=0.9,
            cameraYaw=90,
            cameraPitch=-40,
            cameraTargetPosition=[0, 0, 0.1],
        )
        # Re-enable rendering.
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)

        self.step_Simulation_func(nsteps=100,timesleep=0.01)
    
    
    def calc_camera_params(self,image_size):
        intrinsics = self._camera_instrinsics

        # Set default camera poses.
        front_position = self._camera_pose
        front_rotation = self._camera_orientation
        front_rotation = self._pybullet_client.getQuaternionFromEuler(front_rotation)
        # Default camera configs.
        zrange = (0.01, 10.0)

        # OpenGL camera settings.
        lookdir = np.float32([0, 0, 1]).reshape(3, 1)
        updir = np.float32([0, -1, 0]).reshape(3, 1)
        rotation = self._pybullet_client.getMatrixFromQuaternion(front_rotation)
        rotm = np.float32(rotation).reshape(3, 3)
        lookdir = (rotm @ lookdir).reshape(-1)
        updir = (rotm @ updir).reshape(-1)
        lookat = front_position + lookdir
        focal_len = intrinsics[0]
        znear, zfar = zrange
        viewm = self._pybullet_client.computeViewMatrix(front_position, lookat, updir)
        fovh = (image_size[0] / 2) / focal_len
        fovh = 180 * np.arctan(fovh) * 2 / np.pi

        # Notes: 1) FOV is vertical FOV 2) aspect must be float
        aspect_ratio = image_size[1] / image_size[0]
        projm = self._pybullet_client.computeProjectionMatrixFOV(
            fovh, aspect_ratio, znear, zfar
        )
        return viewm, projm, front_position, lookat, updir
    
    
    def render_and_save_image (self,step_index):
        os.makedirs(self.image_dir, exist_ok=True)
        # Compute camera matrices
        viewm, projm, _, _, _  = self.calc_camera_params(self.image_size)
        # Render image
        _, _, color, _, _ = self._pybullet_client.getCameraImage(
            width=self.image_size[1],
            height=self.image_size[0],
            viewMatrix=viewm,
            projectionMatrix=projm,
            renderer=self._pybullet_client.ER_BULLET_HARDWARE_OPENGL)

        # Format image
        color = np.array(color, dtype=np.uint8).reshape((self.image_size[0], self.image_size[1], 4))[:, :, :3]
        im = Image.fromarray(color)
        im.save(os.path.join(self.image_dir, f"step_{step_index:04d}.png"))
        """ 
        how to use the render_and_save_image: 
        ...
        robot.set_the_fingers_open_close(opening_width)
        for _ in range(50):
            p.stepSimulation()
            render_and_save_image(step_counter)
            step_counter += 1
            time.sleep(1 / 240.0)
        """
        return im


    def show_camera_img(self,image_size):
        viewm, projm, _, _, _ = self.calc_camera_params(image_size)
        _, _, color, _, _ = self._pybullet_client.getCameraImage(
            width=image_size[1],
            height=image_size[0],
            viewMatrix=viewm,
            projectionMatrix=projm,
            flags=self._pybullet_client.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=self._pybullet_client.ER_BULLET_HARDWARE_OPENGL,
        )
        # Get color image.
        color_image_size = (image_size[0], image_size[1], 4)
        color = np.array(color, dtype=np.uint8).reshape(color_image_size)
        color = color[:, :, :3]  # remove alpha channel

        image =  color.astype(np.uint8)   
        plt.imshow(image)
        plt.title("Simulated Camera View")
        plt.axis("off")
        plt.show()
    
    
    def save_state(self):
        self._saved_state = self._pybullet_client.saveState()
   
    def seed(self, seed=None):
        self._rng = np.random.RandomState(seed=seed) 
    
    def _set_robot_target_effector_pose(self, pose,force):
        self._target_effector_pose = pose
        self._robot.set_target_effector_pose(pose,force)
        
    def _compute_state(self):
        
        block_position = self._pybullet_client.getBasePositionAndOrientation(self._block_id)[0]
        block_orientation = self._pybullet_client.getBasePositionAndOrientation(self._block_id)[1]
        block_pose = Pose3d(
            rotation=transform.Rotation.from_quat(block_orientation),
            translation=block_position,)
        
        target_position = self._pybullet_client.getBasePositionAndOrientation(self._target_id)[0]
        target_orientation = self._pybullet_client.getBasePositionAndOrientation(self._target_id)[1]
        target_pose = Pose3d(
            rotation=transform.Rotation.from_quat(target_orientation),
            translation=target_position,)
        
        
        def _yaw_from_pose(pose): # special for  block
            return np.array([pose.rotation.as_euler("xyz", degrees=False)[-1]])
        
        def _yaw_from_pose_robot(pose): # special for the robot
            return np.array([pose.orientation_left.as_euler("xyz", degrees=False)[-1],
                            pose.orientation_right.as_euler("xyz", degrees=False)[-1]])
            
        robot_pose = self._robot.forward_kinematics()
        
        _target_effector_pose_trans_left = self._target_effector_pose.translation_left[0:3]
        _target_effector_pose_translation = _target_effector_pose_trans_left +  self.offset

        _target_pose_trans_left=self._target_pose.translation_left[0:3]
        _target_pose_translation = _target_pose_trans_left + self.offset
                
        gripper_translation_left=robot_pose.translation_left[0:3]
        effector_translation = gripper_translation_left + self.offset
        
        head_gripper_orientation = self._robot.get_joint_state(joint_idx=6) # translation of the head of the gripepr, from -2.9 to 2.9
        
        
        
        obs = collections.OrderedDict(
            block_translation=block_pose.translation[0:3],
            block_orientation=_yaw_from_pose(block_pose),
            
            effector_translation = effector_translation,
            effector_orientation = head_gripper_orientation, # translation of the head of the gripper == the orientation of th fingers 
            
            effector_target_translation=_target_effector_pose_translation,
            
            target_translation=_target_pose_translation,
            target_orientation = _yaw_from_pose(target_pose),
        )
        if self._image_size is not None:
            obs["rgb"] = self.show_camera_img(self._image_size)
        return obs
    
    def _create_observation_space(self, image_size):
        pi2 = math.pi * 2

        obs_dict = collections.OrderedDict(
            block_translation=spaces.Box(low=-5, high=5, shape=(3,)),  # x,y,z
            
            block_orientation=spaces.Box(low=-pi2, high=pi2, shape=(1,)),  # phi
            
            effector_translation = spaces.Box(
                low=np.array([self.workspace_bounds[0, 0] - 0.1, 
                              self.workspace_bounds[0, 1] - 0.1, 0.0]),
                high=np.array([self.workspace_bounds[1, 0] + 0.1, 
                               self.workspace_bounds[1, 1] + 0.1, 0.2]),
            ),

            effector_orientation = spaces.Box(
                low=-pi2,
                high=pi2,
                shape=(1,),
            ),
            
            effector_target_translation=spaces.Box(   
              low=np.array([self.workspace_bounds[0, 0] - 0.1, 
                              self.workspace_bounds[0, 1] - 0.1, 0.0]),
                high=np.array([self.workspace_bounds[1, 0] + 0.1, 
                               self.workspace_bounds[1, 1] + 0.1, 0.2]),
            ),
            target_translation=spaces.Box(low=-5, high=5, shape=(3,)),  # x,y,z
            target_orientation=spaces.Box(
                low=-pi2,
                high=pi2,
                shape=(1,),
            ),
        )
        if image_size is not None:
            obs_dict["rgb"] = spaces.Box(
                low=0, high=255, shape=(image_size[0], image_size[1], 3), dtype=np.uint8
            )
        return spaces.Dict(obs_dict)
    
    def reset(self,reset_poses= True):
        self._pybullet_client.restoreState(self._saved_state)
        
        if reset_poses: 
            
            # Reset the _target_effector_pose
            # (The pose the robot is trying to reach to) :
            orientation = transform.Rotation.from_rotvec([0, math.pi, 0])
            target_center = np.array([0.3, 0.4, 0.5])
            new_translation_left = target_center - self.offset
            new_translation_right = target_center + self.offset
            starting_pose = Pose3d_gripper(translation_left=new_translation_left,
                                    translation_right=new_translation_right,
                                    orientation=orientation) 
            force = 7
            self._set_robot_target_effector_pose(starting_pose, force)
            
            # Reset the block pose: 
            block_x = self.workspace_center_x + self._rng.uniform(low=-0.1, high=0.1)
            block_y = -0.2 + self._rng.uniform(low=-0.15, high=0.15)
            block_translation = np.array([block_x, block_y, 0])
            self.block_translation = block_translation
            block_sampled_angle = self._rng.uniform(math.pi)
            angle_rad = (math.pi / 4 ) # manual degree
            block_rotation = transform.Rotation.from_rotvec([0, 0, angle_rad])
            angle_deg = math.degrees(angle_rad)
            print("block_rotation in radian",angle_rad, "\n", "in degree",angle_deg )
            self._pybullet_client.resetBasePositionAndOrientation(
                self._block_id,
                block_translation.tolist(),
                block_rotation.as_quat().tolist(),)
            
            print(f"\n\nhere is the block translation: {block_translation} ")
            
            # Reset _target_pose
            # the ultimate target (the flat target) pose:
            target_x = self.workspace_center_x + self._rng.uniform(low=-0.10, high=0.10)
            target_y = 0.2 + self._rng.uniform(low=-0.15, high=0.15)
            target_translation = np.array([target_x, target_y, 0.020])

            target_sampled_angle = math.pi + self._rng.uniform(  
                low=-math.pi / 6, high=math.pi / 6)
            target_rotation = transform.Rotation.from_rotvec(
                [0, 0, target_sampled_angle])

            self._pybullet_client.resetBasePositionAndOrientation(
                self._target_id,
                target_translation.tolist(),
                target_rotation.as_quat().tolist(),)
        else: 
            (target_translation,target_orientation_quat,
            ) = self._pybullet_client.getBasePositionAndOrientation(self._target_id)
            target_rotation = transform.Rotation.from_quat(target_orientation_quat)
            target_translation = np.array(target_translation)
        
        print(f"\n\nhere is the target translation: {target_translation} ")


        
        new_translation_left = target_translation - self.offset
        new_translation_right = target_translation + self.offset
        self._target_pose = Pose3d_gripper(translation_left=new_translation_left,
                                translation_right=new_translation_right,
                                orientation=target_rotation) 
        
        if reset_poses:
            self.step_Simulation_func()
            
        state = self._compute_state()
        self._previous_state = state
        
        self._init_goal_distance = self._compute_goal_distance(state)  #TODO do we need it ?
        init_goal_eps = 1e-7
        assert self._init_goal_distance > init_goal_eps
        self.best_fraction_reduced_goal_dist = 0.0

        return state
    
    def _compute_reach_target(self, state):
        xy_block = state["block_translation"]
        xy_target = state["target_translation"]

        xy_block_to_target = xy_target - xy_block
        xy_dir_block_to_target = (xy_block_to_target) / np.linalg.norm(
            xy_block_to_target
        )
        self.reach_target_translation = xy_block + -1 * xy_dir_block_to_target * 0.05
        
    def get_goal_translation(self):
        if self._is_grasped==True: # If the robot is holding the object, the goal is the place target.
            translation_left = self._target_pose.translation_left
            target_translation = translation_left+self.offset
            return target_translation 
        else:   #If not, the goal is to go to the object's current position (pick).
            block_pos, _ = self._pybullet_client.getBasePositionAndOrientation(self._block_id)
            return np.array(block_pos)

    
    def _set_is_grasped(self):
        left_pos = self._pybullet_client.getLinkState(self._robot.gripperarm, self._robot.left_finger)[0]
        right_pos = self._pybullet_client.getLinkState(self._robot.gripperarm, self._robot.right_finger)[0]

        left_pos = np.array(left_pos)
        right_pos = np.array(right_pos)

        distance = np.linalg.norm(left_pos - right_pos)
        if distance <= self._robot.closing_width:
            self._is_grasped = True
        else: 
            self._is_grasped = False
    
    
    def _compute_goal_distance(self, state):
        goal_translation = self.get_goal_translation()
        goal_distance = np.linalg.norm(state["effector_translation"] - goal_translation[0:3])
        return goal_distance
    
    
    def step(self, action):
        p_state = self._compute_state()
        self._set_is_grasped()
        move_to_position = np.array(action)
        target_block_pos = np.r_[p_state["block_translation"][:2], self.effector_height]
        target_place_pos = np.r_[p_state["target_translation"][:2], self.effector_height]
        # Case 1: Move toward the block to pick
        if np.allclose(move_to_position, target_block_pos):
            target_block_pos = np.array(p_state["block_translation"])
            target_block_ori = p_state["block_orientation"] # in radian
            target_place_pos = np.array(p_state["target_translation"])
            target_place_ori = p_state["target_orientation"]# in radian
            
            print("\ 1block_translation", p_state["block_translation"])
            print("effector_translation", np.array(p_state["effector_translation"]))
            
            force = 2
            f_target_block = target_block_pos + np.array([0, 0, 0.1])
            self._robot.move_gripper_to_target( f_target_block, target_block_ori, force)
            for _ in range(50):
                time.sleep(1 / 240.0)
                self._pybullet_client.stepSimulation()
                time.sleep(1 / 240.0)
            n_state = self._compute_state()
            print(" 2effector_translation", np.array(n_state["effector_translation"]))
            print("\n\n Difference in Positions", f_target_block - np.array(n_state["effector_translation"]), "\n\n")
            
            # self._robot.set_target_pick_the_block_2(target_block_pos, target_block_ori)
            # self._robot.set_target_place_the_block (target_place_pos, target_block_ori)


        # Case 2: Move toward the target to place
        elif np.allclose(move_to_position, target_place_pos) and self._is_grasped:
            target_place_pos = np.array([p_state["target_translation"]])
            target_place_ori = p_state["target_orientation"]
            self._robot.set_target_place_the_block(target_place_pos, target_place_ori)

        # Case 3: General movement 
        else:
            force = 7
            orientation_target = p_state["block_orientation"]
            self._robot.move_gripper_to_target(move_to_position,orientation_target, force)
            for _ in range(200):
                time.sleep(1 / 50)
                self._pybullet_client.stepSimulation()
                time.sleep(1 / 240.0)

        info = {}
        info["block_translation"] = p_state["block_translation"]
        info["grasped"] = self._is_grasped
        info["distance_to_target"] = np.linalg.norm(p_state["block_translation"][:2] - p_state["target_translation"][:2])
        
        state = self._compute_state()
        goal_distance = self._compute_goal_distance(state)
        fraction_reduced_goal_distance = 1.0 - (goal_distance / self._init_goal_distance)

        if fraction_reduced_goal_distance > self.best_fraction_reduced_goal_dist:
            self.best_fraction_reduced_goal_dist = fraction_reduced_goal_distance

        reward = self.best_fraction_reduced_goal_dist
        done = False

        if goal_distance < self.goal_dist_tolerance:
            reward = 1.0
            done = True

        return state, reward, done, {}
    
    @property
    def succeeded(self):
        state = self._compute_state()
        goal_distance = self._compute_goal_distance(state)
        if goal_distance < self.goal_dist_tolerance:
            return True
        return False
    
    @property
    def goal_distance(self):
        state = self._compute_state()
        return self._compute_goal_distance(state)

    def render(self, mode="rgb_array"):
        if self._image_size is not None:
            image_size = self._image_size
        else:
            # This allows rendering even for state-only obs,
            # for visualization.
            image_size = (IMAGE_HEIGHT, IMAGE_WIDTH)

        data = self.render_and_save_image(step_index=10)
        if mode == "human":
            if self.rendered_img is None:
                self.rendered_img = plt.imshow(
                    np.zeros((image_size[0], image_size[1], 4))
                )
            else:
                self.rendered_img.set_data(data)
            plt.draw()
            plt.pause(0.00001)
        return data

    def close(self):
        self._pybullet_client.disconnect()



if "BlockPick-v0" in registration.registry:
    del registration.registry["BlockPick-v0"]


registration.register(
    id="BlockPick-v0",
    entry_point=BlockPick,
    max_episode_steps=50,
)
