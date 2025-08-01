import enum
import math
import collections
from gym import spaces
from gym.envs import registration
from typing import Dict, List, Optional, Tuple, Union
from block_pushing.utils.pose3d_gripper import Pose3d_gripper
from diffusion_policy.env.block_pushing.utils.pose3d import Pose3d
from block_pushing.utils.franka_panda_sim_robot import GripperArmSimRobot
from block_pushing.utils import utils_pybullet, franka_panda_sim_robot
from block_pushing.utils.utils_pybullet import ObjState, XarmState
import pybullet_utils.bullet_client as bullet_client
from scipy.spatial import transform
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pybullet
import gym
import time
import os


X_MIN_REAL = 0.15
X_MAX_REAL = 0.6
Y_MIN_REAL = -0.3048
Y_MAX_REAL = 0.3048
DEFAULT_CAMERA_POSE = (1.1, 0, 0.75)
DEFAULT_CAMERA_ORIENTATION = (np.pi / 4, np.pi, -np.pi / 2)
IMAGE_WIDTH = 320
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
BLOCK_URDF_PATH = "third_party/py/envs/assets/block.urdf"
PLANE_URDF_PATH = "third_party/bullet/examples/pybullet/gym/pybullet_data/" "plane.urdf"
WORKSPACE_URDF_PATH = "third_party/py/envs/assets/workspace.urdf"
ZONE_URDF_PATH = "third_party/py/envs/assets/zone.urdf"
INSERT_URDF_PATH = "third_party/py/envs/assets/insert.urdf"
INITIAL_JOINT_POSITIONS = np.array(
    [
        0.0, 
        -0.5235987755982988, 
        0.0, -1.0471975511965976, 
        0.0, 1.5707963267948966, 
        0.0, 
        0.0, 
        0.0
    ])

class BlockPick(gym.Env):
    def __init__(
    self,
    control_frequency=10.0,
    image_size=None,
    shared_memory=False, 
    seed=None,
    goal_dist_tolerance=0.01,
    effector_height=None,
    visuals_mode="default",
    abs_action=False
):
      
        # Mimic RealSense D415 camera parameters.
        if visuals_mode == "default":
            _camera_pose = DEFAULT_CAMERA_POSE
            _camera_orientation = DEFAULT_CAMERA_ORIENTATION
            workspace_bounds = WORKSPACE_BOUNDS
            _camera_instrinsics = CAMERA_INTRINSICS
            _workspace_urdf_path = WORKSPACE_URDF_PATH
        else:
            _camera_pose = CAMERA_POSE_REAL
            _camera_orientation = CAMERA_ORIENTATION_REAL
            workspace_bounds = WORKSPACE_BOUNDS_REAL
            _camera_instrinsics = CAMERA_INTRINSICS_REAL
            _workspace_urdf_path = WORKSPACE_URDF_PATH_REAL
        
        self._pybullet_client = None
        self._connection_mode = pybullet.DIRECT
        if shared_memory:
            self._connection_mode = pybullet.SHARED_MEMORY
        self._setup_the_scene()
        self.image_size = np.array([320, 240])
        self.image_dir="trajectory_images"
        self.save_state()
        self.reset()
        self._target_effector_pose = None
        self._target_pose = None
        self._control_frequency = None
        assert isinstance(self._pybullet_client, bullet_client.BulletClient)


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
        bullet_client.BulletClient(connection_mode=self._connection_mode)
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
    
    
    def _set_robot_target_effector_pose(self, pose):
        self._target_effector_pose = pose
        self._robot.set_target_effector_pose(pose)
        
    def _compute_state(self):
        
        block_position = self._pybullet_client.getBasePositionAndOrientation(self._block_id)[0]
        block_orientation = self._pybullet_client.getBasePositionAndOrientation(self._block_id)[1]
        block_pose = Pose3d(
            rotation=transform.Rotation.from_quat(block_orientation),
            translation=block_position,)
        
        def _yaw_from_pose(pose): # special for  block
            return np.array([pose.rotation.as_euler("xyz", degrees=False)[-1]])
        
        def _yaw_from_pose_robot(pose): # special for the robot
            return np.array([pose.orientation_left.as_euler("xyz", degrees=False)[-1],
                            pose.orientation_right.as_euler("xyz", degrees=False)[-1]])
            
        robot_pose = self._robot.forward_kinematics()
        
        
        obs = collections.OrderedDict(
            block_translation=block_pose.translation[0:3],
            block_orientation=_yaw_from_pose(block_pose),
            
            gripper_translation_left=robot_pose.translation_left[0:3],
            gripper_translation_right=robot_pose.translation_right[0:3],
            
            effector_target_translation=self._target_effector_pose.translation[0:2],
            
            target_translation=self._target_pose.translation[0:3],
        )
        if self._image_size is not None:
            obs["rgb"] = self._render_camera(self._image_size)
        return obs