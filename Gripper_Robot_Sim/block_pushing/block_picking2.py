
import collections
import enum
import math
import time
from typing import Dict, List, Optional, Tuple, Union

import gym
from gym import spaces
from gym.envs import registration
from block_pushing.utils import utils_pybullet
from block_pushing.utils import franka_panda_sim_robot
# from block_pushing.utils import xarm_sim_robot
# from block_pushing.utils.pose3d import Pose3d
from block_pushing.utils.pose3d import Pose3d_gripper
from block_pushing.utils.utils_pybullet import ObjState
from block_pushing.utils.utils_pybullet import XarmState
import numpy as np
from scipy.spatial import transform
import pybullet
import pybullet_utils.bullet_client as bullet_client

import matplotlib.pyplot as plt

BLOCK_URDF_PATH = "third_party/py/envs/assets/block.urdf"
PLANE_URDF_PATH = "third_party/bullet/examples/pybullet/gym/pybullet_data/" "plane.urdf"
WORKSPACE_URDF_PATH = "third_party/py/envs/assets/workspace.urdf"
ZONE_URDF_PATH = "third_party/py/envs/assets/zone.urdf"
INSERT_URDF_PATH = "third_party/py/envs/assets/insert.urdf"

EFFECTOR_HEIGHT = 0.06
EFFECTOR_DOWN_ROTATION = transform.Rotation.from_rotvec([0, math.pi, 0])

WORKSPACE_BOUNDS = np.array(((0.15, -0.5), (0.7, 0.5)))

# Min/max bounds calculated from oracle data using:
# ibc/environments/board2d_dataset_statistics.ipynb
# to calculate [mean - 3 * std, mean + 3 * std] using the oracle data.
# pylint: disable=line-too-long
ACTION_MIN = np.array([-0.02547718, -0.02090043], np.float32)
ACTION_MAX = np.array([0.02869084, 0.04272365], np.float32)
EFFECTOR_TARGET_TRANSLATION_MIN = np.array(
    [0.1774151772260666, -0.6287994794547558], np.float32
)
EFFECTOR_TARGET_TRANSLATION_MAX = np.array(
    [0.5654461532831192, 0.5441607423126698], np.float32
)
EFFECTOR_TARGET_TO_BLOCK_TRANSLATION_MIN = np.array(
    [-0.07369826920330524, -0.11395704373717308], np.float32
)
EFFECTOR_TARGET_TO_BLOCK_TRANSLATION_MAX = np.array(
    [0.10131562314927578, 0.19391131028532982], np.float32
)
EFFECTOR_TARGET_TO_TARGET_TRANSLATION_MIN = np.array(
    [-0.17813862301409245, -0.3309651017189026], np.float32
)
EFFECTOR_TARGET_TO_TARGET_TRANSLATION_MAX = np.array(
    [0.23726161383092403, 0.8404090404510498], np.float32
)
BLOCK_ORIENTATION_COS_SIN_MIN = np.array(
    [-2.0649861991405487, -0.6154364347457886], np.float32
)
BLOCK_ORIENTATION_COS_SIN_MAX = np.array(
    [1.6590178310871124, 1.8811014890670776], np.float32
)
TARGET_ORIENTATION_COS_SIN_MIN = np.array(
    [-1.0761439241468906, -0.8846937336493284], np.float32
)
TARGET_ORIENTATION_COS_SIN_MAX = np.array(
    [-0.8344330154359341, 0.8786859593819827], np.float32
)

# Hardcoded Pose joints to make sure we don't have surprises from using the
# IK solver on reset. The joint poses correspond to the Pose with:
#   rotation = rotation3.Rotation3.from_axis_angle([0, 1, 0], math.pi)
#   translation = np.array([0.3, -0.4, 0.07])
INITIAL_JOINT_POSITIONS = np.array(
    [
        -0.9254632489674508,
        0.6990770671568564,
        -1.106629064060494,
        0.0006653351931553931,
        0.3987969742311386,
        -4.063402065624296,
    ]
)

DEFAULT_CAMERA_POSE = (1.0, 0, 0.75)
DEFAULT_CAMERA_ORIENTATION = (np.pi / 4, np.pi, -np.pi / 2)
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240
CAMERA_INTRINSICS = (
    0.803 * IMAGE_WIDTH,  # fx
    0,
    IMAGE_WIDTH / 2.0,  # cx
    0,
    0.803 * IMAGE_WIDTH,  # fy
    IMAGE_HEIGHT / 2.0,  # cy
    0,
    0,
    1,
)

# "Realistic" visuals.
X_MIN_REAL = 0.15
X_MAX_REAL = 0.6
Y_MIN_REAL = -0.3048
Y_MAX_REAL = 0.3048
WORKSPACE_BOUNDS_REAL = np.array(((X_MIN_REAL, Y_MIN_REAL), (X_MAX_REAL, Y_MAX_REAL)))
WORKSPACE_URDF_PATH_REAL = "third_party/py/ibc/environments/assets/workspace_real.urdf"
CAMERA_POSE_REAL = (0.75, 0, 0.5)
CAMERA_ORIENTATION_REAL = (np.pi / 5, np.pi, -np.pi / 2)

IMAGE_WIDTH_REAL = 320
IMAGE_HEIGHT_REAL = 180
CAMERA_INTRINSICS_REAL = (
    0.803 * IMAGE_WIDTH_REAL,  # fx
    0,
    IMAGE_WIDTH_REAL / 2.0,  # cx
    0,
    0.803 * IMAGE_WIDTH_REAL,  # fy
    IMAGE_HEIGHT_REAL / 2.0,  # cy
    0,
    0,
    1,
)


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
        # Init camera, workspace, physics, visuals
        self._setup_pybullet()
        self._load_workspace()
        self._load_robot()  # Use GripperArmSimRobot
        self._load_objects()
        self._define_spaces()
        
        
        self._visuals_mode = visuals_mode
        if visuals_mode == "default":
            self._camera_pose = DEFAULT_CAMERA_POSE
            self._camera_orientation = DEFAULT_CAMERA_ORIENTATION
            self.workspace_bounds = WORKSPACE_BOUNDS
            self._image_size = image_size
            self._camera_instrinsics = CAMERA_INTRINSICS
            self._workspace_urdf_path = WORKSPACE_URDF_PATH
        else:
            self._camera_pose = CAMERA_POSE_REAL
            self._camera_orientation = CAMERA_ORIENTATION_REAL
            self.workspace_bounds = WORKSPACE_BOUNDS_REAL
            self._image_size = image_size
            self._camera_instrinsics = CAMERA_INTRINSICS_REAL
            self._workspace_urdf_path = WORKSPACE_URDF_PATH_REAL
            
            
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,))  # x, y
        self.observation_space = self._create_observation_space(image_size)

        self._control_frequency = control_frequency   # If control_frequency = 10.0, the robot receives an action every 0.1 seconds
        
        self._step_frequency = (     # its the inverse of PyBullet’s internal fixedTimeSte  #Example: If fixedTimeStep = 1/240, then step_frequency = 240.0. This means PyBullet simulates physics at 240 Hz.
            1 / self._pybullet_client.getPhysicsEngineParameters()["fixedTimeStep"])

        if self._step_frequency % self._control_frequency != 0:
            raise ValueError(
                "Control frequency should be a multiple of the "
                "configured Bullet TimeStep.")
            
        # Use saved_state and restore to make reset safe as no simulation state has
        # been updated at this state, but the assets are now loaded.
        self.save_state()
        self.reset()


    def _setup_workspace_and_robot(self):
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

        self._robot = franka_panda_sim_robot.GripperArmSimRobot(  # Khodam
            self._pybullet_client,
            initial_joint_positions=INITIAL_JOINT_POSITIONS, #Khodam
            color="white" if self._visuals_mode == "real" else "default",)


            
    def _setup_pybullet(self):
        # Connect to pybullet (DIRECT or GUI)
        pass

    def _load_workspace(self):
        # Load table, plane, zone
        pass

    def _load_robot(self):
        # Instantiate GripperArmSimRobot
        pass

    def _load_objects(self):
        # Add blocks to the workspace
        # Maybe randomly scatter 1 or more blocks
        pass

    def _define_spaces(self):
        # Define observation_space and action_space
        # action: [dx, dy, dz, gripper_action]
        pass

    def reset(self):
        # Reset pybullet state
        # Reset block + target + robot to initial positions
        # Return initial observation
        pass

    def step(self, action):
        # Parse action → pose + gripper state
        # Apply inverse kinematics (IK)
        # Move robot and step simulation
        # Update internal state
        # Compute reward, done
        # Return obs, reward, done, info
        pass

    def _compute_state(self):
        # Get block pose, gripper pose, maybe gripper state
        pass

    def render(self, mode="rgb_array"):
        # Optionally render camera image using pybullet
        pass

    def close(self):
        # Disconnect from pybullet
        pass
