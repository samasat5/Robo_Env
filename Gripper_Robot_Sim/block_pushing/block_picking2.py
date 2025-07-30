
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
from block_pushing.utils.utils_pybullet import GripperArmSimRobot

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
import logging
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

# When resetting multiple targets, they should all be this far apart.
MIN_BLOCK_DIST = 0.1
MIN_TARGET_DIST = 0.12
# pylint: enable=line-too-long
NUM_RESET_ATTEMPTS = 1000

# Random movement of blocks
RANDOM_X_SHIFT = 0.1
RANDOM_Y_SHIFT = 0.15

logging.basicConfig(
    level="INFO",
    format="%(asctime)s [%(levelname)s] %(message)s",
    filemode="w",
)
logger = logging.getLogger()

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
            

        self._rng = np.random.RandomState(seed=seed)
        self._block_ids = None
        self._previous_state = None
        self._robot = None
        self._workspace_uid = None
        self._target_id = None
        self._target_pose = None
        self._target_effector_pose = None
        self._pybullet_client = None
        self.reach_target_translation = None
        self._setup_pybullet_scene()
        self._saved_state = None

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
        
        
    def step_Simulation_func(self, nsteps=100):
        for _ in range(nsteps):
            self._pybullet_client.stepSimulation()


    def _setup_the_scene(self):
        
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
        
        self._target_id = utils_pybullet.load_urdf(
            self._pybullet_client, ZONE_URDF_PATH, useFixedBase=True)
        
        self._block_ids = [
            utils_pybullet.load_urdf(
                self._pybullet_client, BLOCK_URDF_PATH, useFixedBase=False)]
        
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

        # Re-enable rendering.
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)

        self.step_Simulation_func(nsteps=100)

    def seed(self, seed=None):
        self._rng = np.random.RandomState(seed=seed)

    # Robot methods
    def _set_robot_target_effector_pose(self, pose):
        self._target_effector_pose = pose
        self._robot.set_target_effector_pose(pose)
        
    def reset(self, reset_poses=True):
        
        workspace_center_x = 0.4

        if reset_poses:
            self._pybullet_client.restoreState(self._saved_state)
            
            center_translation = np.array([0.3, -0.4, self.effector_height])
            center_rotation = EFFECTOR_DOWN_ROTATION
            starting_pose = self._compute_pose(center_translation,center_rotation)
            self._set_robot_target_effector_pose(starting_pose)


            # Reset block pose.
            block_x = workspace_center_x + self._rng.uniform(low=-0.1, high=0.1)
            block_y = -0.2 + self._rng.uniform(low=-0.15, high=0.15)
            block_translation = np.array([block_x, block_y, 0])
            block_sampled_angle = self._rng.uniform(math.pi)
            
            block_rotation = transform.Rotation.from_rotvec([0, 0, block_sampled_angle])

            self._pybullet_client.resetBasePositionAndOrientation(
                self._block_ids[0],
                block_translation.tolist(),
                block_rotation.as_quat().tolist(),
            )

            # Reset target pose.
            target_x = workspace_center_x + self._rng.uniform(low=-0.10, high=0.10)
            target_y = 0.2 + self._rng.uniform(low=-0.15, high=0.15)
            target_translation = np.array([target_x, target_y, 0.020])

            target_sampled_angle = math.pi + self._rng.uniform(
                low=-math.pi / 6, high=math.pi / 6
            )
            target_rotation = transform.Rotation.from_rotvec(
                [0, 0, target_sampled_angle]
            )

            self._pybullet_client.resetBasePositionAndOrientation(
                self._target_id,
                target_translation.tolist(),
                target_rotation.as_quat().tolist(),
            )
        else:
            (target_translation,
            target_orientation_quat,) = self._pybullet_client.getBasePositionAndOrientation(self._target_id)
            
            target_center_rotation = transform.Rotation.from_quat(target_orientation_quat)
            target_center_translationt = np.array(target_translation)
            finger_offset = 0.02  # 2 cm on each side in y-axis
            translation_left = target_center_translationt + np.array([0.0, -finger_offset, 0.0])
            translation_right = target_center_translationt + np.array([0.0, finger_offset, 0.0])
            
        self._target_pose = Pose3d_gripper(translation_left, translation_right, target_center_rotation, target_center_rotation)

        if reset_poses:
            self.step_Simulation_func()

        state = self._compute_state()
        self._previous_state = state

        self._init_goal_distance = self._compute_goal_distance(state)
        init_goal_eps = 1e-7
        assert self._init_goal_distance > init_goal_eps
        self.best_fraction_reduced_goal_dist = 0.0

        return state
    
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
        self._step_robot_and_sim(action)

        state = self._compute_state()
        done = False
        reward = self._get_reward(state)
        if reward >= 0.5:
            # Terminate the episode if both blocks are close enough to the targets.
            done = True

        info = self._event_manager.get_info()
        return state, reward, done, info

    def _compute_state(self):
        effector_pose = self._robot.forward_kinematics()
        block_position_and_orientation = (self._pybullet_client.getBasePositionAndOrientation(self._block_ids[0]))            
        rotation_left = transform.Rotation.from_quat(block_position_and_orientation[1])
        rotation_right = transform.Rotation.from_quat(block_position_and_orientation[1])
    
        center_translation = block_position_and_orientation[0]
        finger_offset = 0.02  # 2 cm on each side in y-axis
        translation_left = center_translation + np.array([0.0, -finger_offset, 0.0])
        translation_right = center_translation + np.array([0.0, finger_offset, 0.0])
        block_pose = Pose3d_gripper(
            rotation_left,
            rotation_right,
            translation_left,
            translation_right) 
        effector_pose = self._robot.forward_kinematics()

        def _yaw_from_pose(pose):
            return np.array([pose.rotation.as_euler("xyz", degrees=False)[-1] % np.pi])

        obs = collections.OrderedDict(
            block_translation=block_pose.translation[0:2],
            block_orientation=_yaw_from_pose(block_pose),
            # block2_translation=block_pose.translation[0:2],
            # block2_orientation=_yaw_from_pose(block_pose),
            effector_translation=effector_pose.translation[0:2],
            effector_target_translation=self._target_effector_pose.translation[0:2],
            target_translation=self._target_poses[0].translation[0:2],
            target_orientation=_yaw_from_pose(self._target_poses[0]),
            # target2_translation=self._target_poses[1].translation[0:2],
            # target2_orientation=_yaw_from_pose(self._target_poses[1]),
        )

        if self._image_size is not None:
            obs["rgb"] = self._render_camera(self._image_size)
        return obs
    
    def _compute_pose(self, center_translation, center_rotation):
        finger_offset = 0.02  # 2 cm on each side in y-axis
        translation_left = center_translation + np.array([0.0, -finger_offset, 0.0])
        translation_right = center_translation + np.array([0.0, finger_offset, 0.0])
        target_effector_pose = Pose3d_gripper(
            center_rotation,
            center_rotation,
            translation_left,
            translation_right) 
        return target_effector_pose
    
    def _step_robot_and_sim(self, action):
        """Steps the robot and pybullet sim."""
        # Compute target_effector_pose by shifting the effector's pose by the action.
    
        target_effector_translation = np.array([action[0], action[1], action[2]])

        target_effector_translation[0:2] = np.clip(
            target_effector_translation[0:2],
            self.workspace_bounds[0],
            self.workspace_bounds[1],
        )
        center_translation = target_effector_translation
        center_rotation = EFFECTOR_DOWN_ROTATION
        
        target_effector_pose = self._compute_pose(center_translation, center_rotation)
        self._set_robot_target_effector_pose(target_effector_pose)

        # Update sleep time dynamically to stay near real-time.
        frame_sleep_time = 0
        if self._connection_mode == pybullet.SHARED_MEMORY:
            cur_time = time.time()
            if self._last_loop_time is not None:
                # Calculate the total, non-sleeping time from the previous frame, this
                # includes the actual step as well as any compute that happens in the
                # caller thread (model inference, etc).
                compute_time = (
                    cur_time
                    - self._last_loop_time
                    - self._last_loop_frame_sleep_time * self._sim_steps_per_step
                )
                # Use this to calculate the current frame's total sleep time to ensure
                # that env.step runs at policy rate. This is an estimate since the
                # previous frame's compute time may not match the current frame.
                total_sleep_time = max((1 / self._control_frequency) - compute_time, 0)
                # Now spread this out over the inner sim steps. This doesn't change
                # control in any way, but makes the animation appear smooth.
                frame_sleep_time = total_sleep_time / self._sim_steps_per_step
            else:
                # No estimate of the previous frame's compute, assume it is zero.
                frame_sleep_time = 1 / self._step_frequency

            # Cache end of this loop time, to compute sleep time on next iteration.
            self._last_loop_time = cur_time
            self._last_loop_frame_sleep_time = frame_sleep_time

        for _ in range(self._sim_steps_per_step):
            if self._connection_mode == pybullet.SHARED_MEMORY:
                self.sleep_spin(frame_sleep_time)
            self._pybullet_client.stepSimulation()
            
    def sleep_spin(sleep_time_sec):
        """Spin wait sleep. Avoids time.sleep accuracy issues on Windows."""
        if sleep_time_sec <= 0:
            return
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < sleep_time_sec:
            pass
       
       
    def _compute_goal_distance(self, state):
        real_distance_target_state = np.linalg.norm(state["target_translation"] - state["block_translation"])  # euclidean distance of block and target on the ground
        target_translation_0 = np.array([state["target_translation"][0], state["target_translation"][1], 0.2])
        assert state["block_translation"].shape == (3,)
        distance_0 = np.linalg.norm(target_translation_0 - state["block_translation"]) # elevated goal # distance that the robot needs to travel for a sucessful grasp-and-place task
        return distance_0

    def _get_reward(self, state):
        # Reward is 1. the block is inside the target
        block_pos = state["block_translation"]
        target_pos = state["target_translation"]

        # Compute 3D distance between block and target
        dist = np.linalg.norm(block_pos - target_pos)

        # Check if block is within goal tolerance
        if dist < self.goal_dist_tolerance:
            logger.info(f"Block reached target on step {self._step_num}")
            self._event_manager.target(step=self._step_num, block_id=0, target_id=0)
            return 1.0
        return 0.0
  
    
    
    @property
    def succeeded(self):
        state = self._compute_state()
        reward = self._get_reward(state)
        if reward >= 0.5:
            return True
        return False
    
    def _create_observation_space(self, image_size):
        pi2 = math.pi * 2

        obs_dict = collections.OrderedDict(
            block_translation=spaces.Box(low=-5, high=5, shape=(2,)),  # x,y
            block_orientation=spaces.Box(low=-pi2, high=pi2, shape=(1,)),  # phi
            block2_translation=spaces.Box(low=-5, high=5, shape=(2,)),  # x,y
            block2_orientation=spaces.Box(low=-pi2, high=pi2, shape=(1,)),  # phi
            effector_translation=spaces.Box(
                low=WORKSPACE_BOUNDS[0] - 0.1,
                high=WORKSPACE_BOUNDS[1] + 0.1,
            ),  # x,y
            effector_target_translation=spaces.Box(
                low=WORKSPACE_BOUNDS[0] - 0.1,
                high=WORKSPACE_BOUNDS[1] + 0.1,
            ),  # x,y
            target_translation=spaces.Box(low=-5, high=5, shape=(2,)),  # x,y
            target_orientation=spaces.Box(
                low=-pi2,
                high=pi2,
                shape=(1,),
            ),  # theta
            target2_translation=spaces.Box(low=-5, high=5, shape=(2,)),  # x,y
            target2_orientation=spaces.Box(
                low=-pi2,
                high=pi2,
                shape=(1,),
            ),  # theta
        )
        if image_size is not None:
            obs_dict["rgb"] = spaces.Box(
                low=0, high=255, shape=(image_size[0], image_size[1], 3), dtype=np.uint8
            )
        return spaces.Dict(obs_dict)
        
    def get_pybullet_state(self):
        """Save pybullet state of the scene.

        Returns:
          dict containing 'robots', 'robot_end_effectors', 'targets', 'objects',
            each containing a list of ObjState.
        """
        state: Dict[str, List[ObjState]] = {}

        state["robots"] = [
            GripperArmSimRobot.get_bullet_state(
                self._pybullet_client,
                self.robot.xarm,
                target_effector_pose=self._target_effector_pose,
                goal_translation=None,
            )
        ]

        state["robot_end_effectors"] = []
        if self.robot.end_effector:
            state["robot_end_effectors"].append(
                ObjState.get_bullet_state(
                    self._pybullet_client, self.robot.end_effector
                )
            )

        state["targets"] = []
        if self._target_ids:
            for target_id in self._target_ids:
                state["targets"].append(
                    ObjState.get_bullet_state(self._pybullet_client, target_id)
                )

        state["objects"] = []
        for obj_id in self.get_obj_ids():
            state["objects"].append(
                ObjState.get_bullet_state(self._pybullet_client, obj_id)
            )

        return state
        
    def set_pybullet_state(self, state):
        """Restore pyullet state.

        WARNING: py_environment wrapper assumes environments aren't reset in their
        constructor and will often reset the environment unintentionally. It is
        always recommended that you call env.reset on the tfagents wrapper before
        playback (replaying pybullet_state).

        Args:
          state: dict containing 'robots', 'robot_end_effectors', 'targets',
            'objects', each containing a list of ObjState.
        """

        assert isinstance(state["robots"][0], GripperArmSimRobot)
        gripperarm_state: GripperArmSimRobot = state["robots"][0]
        gripperarm_state.set_bullet_state(self._pybullet_client, self.robot.gripperarm)
        self._set_robot_target_effector_pose(gripperarm_state.target_effector_pose)

        def _set_state_safe(obj_state, obj_id):
            if obj_state is not None:
                assert obj_id is not None, "Cannot set state for missing object."
                obj_state.set_bullet_state(self._pybullet_client, obj_id)
            else:
                assert obj_id is None, f"No state found for obj_id {obj_id}"

        robot_end_effectors = state["robot_end_effectors"]
        _set_state_safe(
            None if not robot_end_effectors else robot_end_effectors[0],
            self.robot.end_effector,
        )

        for target_state, target_id in zip(state["targets"], self._target_ids):
            _set_state_safe(target_state, target_id)

        obj_ids = self.get_obj_ids()
        assert len(state["objects"]) == len(obj_ids), "State length mismatch"
        for obj_state, obj_id in zip(state["objects"], obj_ids):
            _set_state_safe(obj_state, obj_id)

        self.reset(reset_poses=False)
        
        
    def render(self, mode="rgb_array"):
        # Optionally render camera image using pybullet
        pass

    def close(self):
        # Disconnect from pybullet
        pass
