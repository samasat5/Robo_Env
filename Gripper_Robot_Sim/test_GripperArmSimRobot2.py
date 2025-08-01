
import pybullet as p
import pybullet_data
import time
import numpy as np
import pdb
import os
from scipy.spatial import transform
from gym import spaces
import collections
import enum
import math
from block_pushing.utils.xarm_sim_robot import XArmSimRobot
from block_pushing.utils.franka_panda_sim_robot import GripperArmSimRobot
from block_pushing.utils import franka_panda_sim_robot
import matplotlib.pyplot as plt
from block_pushing.utils.pose3d import Pose3d
from block_pushing.utils.pose3d_gripper import Pose3d_gripper

from block_pushing.utils import utils_pybullet
from scipy.spatial.transform import Rotation

# Start PyBullet in GUI mode
import pybullet as p
import pybullet_data
import time
import numpy as np
from scipy.spatial.transform import Rotation



# client = p.connect(p.DIRECT)
# p.setGravity(0, 0, -9.81)
# p.setAdditionalSearchPath(pybullet_data.getDataPath())
# p.loadURDF("plane.urdf",[0, 0, -0.001])

# p.resetDebugVisualizerCamera(
#     cameraDistance=1,
#     cameraYaw=100,
#     cameraPitch=-30,
#     cameraTargetPosition=[0, 0, 0.1],
# )

# block = p.loadURDF("block2.urdf",[0.2, 0.5, 0.01]  , useFixedBase = False ) # size="0.04 0.04 0.04
# workspace = p.loadURDF("workspace.urdf",[0.35, 0, 0.0]  , useFixedBase = True )
# zone = p.loadURDF("zone.urdf",[0.4, 0.4, 0.1]  , useFixedBase = False ) # scale="0.006 0.006 0.00005" : the size is 0.006 x 20 = 0.12, origin "-0.01 0 0.02"
# # Fixing the zone on the workspace
# p.createConstraint(   
#     parentBodyUniqueId=workspace,
#     parentLinkIndex=-1,                # -1 means the base link of the parent
#     childBodyUniqueId=zone ,
#     childLinkIndex=-1,                 # -1 means the base link of the child
#     jointType=p.JOINT_FIXED,
#     jointAxis=[0, 0, 0],
#     parentFramePosition=[0, 0, 0.02],   # Position of zone relative to workspace
#     childFramePosition=[0, 0, 0],      # Position of zone relative to its own origin
# )
# # blue_cube = p.loadURDF("blue_cube.urdf",[0.2, 0, 0.0]  , useFixedBase = False )
# # red_moon = p.loadURDF("red_moon.urdf",[0.4, 0, 0.0]  , useFixedBase = False )

# # Create robot with end effector
# robot = GripperArmSimRobot(pybullet_client=p)

# # Reset joints to home position
# print("\n[TEST] reset_joints")
# robot.reset_joints(robot.initial_joint_positions)
# # Step simulation
# for _ in range(100):
#     p.stepSimulation()
#     time.sleep(1 / 240.0)
# time.sleep(2)


# # Get joint positions, velocities, torques
# print("\n[TEST] get_joints_measured")
# positions, velocities, torques = robot.get_joints_measured()
# print("Positions:", positions)
# print("Velocities:", velocities)
# print("Torques:", torques)



# # Test FK
# print("\n[TEST] forward_kinematics")## uses getLinkState
# pose = robot.forward_kinematics()  # computes the current 3D pose (position + orientation) of the robot's end effector in  
# print("FK pose translation left:", pose.translation_left) # x,y,z
# print("FK pose rotation left (quat):", pose.rotation_left.as_quat()) #quaternion [x, y, z, w]

# print("robot current state",robot._get_current_translation_orientation())

# # test Inverse Kinematics
# print("\n[TEST] inverse_kinematics")
# # new_translation_left = pose.translation_left + np.array([0, 0, -1]) 
# # new_translation_right = pose.translation_right + np.array([0, 0, -1])  
# target_center = np.array([0.2, 0.5, 0.01 + 0.015])
# offset = np.array([0.03, 0, 0])  # assume fingers are 6cm apart
# new_translation_left = target_center - offset
# new_translation_right = target_center + offset
 
# new_pose = Pose3d_gripper(translation_left=new_translation_left,
#                           translation_right=new_translation_right,
#                           rotation_left=pose.rotation_left, 
#                           rotation_right=pose.rotation_left) #Create a new Pose3d with same orientation but new position
# ik_solution = robot.inverse_kinematics(new_pose) #Solve Inverse Kinematics to find joint angles to reach new_pose
# print("IK Joint Angles (target_joint_positions):", ik_solution)





# # Apply pose via IK (going toward the block and grasping it)
# print("\n[TEST] set_target_effector_pose")
# size_of_the_block = 0.04
# opening_width = size_of_the_block + 0.0001 # grabbing size to grasp the block
# robot.set_the_fingers_open_close(opening_width)
# for _ in range(50):
#     p.stepSimulation()
#     time.sleep(1 / 240.0)
# force = 7
# robot.set_target_effector_pose(new_pose,force)
# for _ in range(100):
#     p.stepSimulation()
#     time.sleep(1 / 240.0)
# closing_width = - 0.0001
# robot.set_the_fingers_open_close(closing_width)
# for _ in range(50):
#     p.stepSimulation()
#     time.sleep(1 / 240.0) 
    

# print("robot current state",robot._get_current_translation())

    
# time.sleep(3)


""" 
self._setup_the_scene():
"""
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
BLOCK_URDF_PATH = "third_party/py/envs/assets/block.urdf"
PLANE_URDF_PATH = "third_party/bullet/examples/pybullet/gym/pybullet_data/" "plane.urdf"
WORKSPACE_URDF_PATH = "third_party/py/envs/assets/workspace.urdf"
ZONE_URDF_PATH = "third_party/py/envs/assets/zone.urdf"
INSERT_URDF_PATH = "third_party/py/envs/assets/insert.urdf"
p.connect(p.GUI)
p.resetSimulation()
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.setPhysicsEngineParameter(enableFileCaching=0)
p.setGravity(0, 0, -9.8)

utils_pybullet.load_urdf(
    p, PLANE_URDF_PATH, basePosition=[0, 0, -0.001])

_workspace_uid = utils_pybullet.load_urdf(
    p,
    WORKSPACE_URDF_PATH,
    basePosition=[0.35, 0, 0.0],)

_robot = franka_panda_sim_robot.GripperArmSimRobot(p,
                                                   INITIAL_JOINT_POSITIONS)
# initial_joint_angles = []
# for link in range(9):
#     pos_link = p.getJointState(_robot.gripperarm, link)[0]
#     initial_joint_angles.append(pos_link)
# print("init:",initial_joint_angles)
    

_target_id = utils_pybullet.load_urdf(p,ZONE_URDF_PATH,
                                      [0.4999, -0.36, 0.1], 
                                      useFixedBase=True)
_block_id = utils_pybullet.load_urdf(p, BLOCK_URDF_PATH, 
                                     [0.2, 0.47, 0.01],
                                     useFixedBase=False)

p.createConstraint(   
    parentBodyUniqueId=_workspace_uid,
    parentLinkIndex=-1,                # -1 means the base link of the parent
    childBodyUniqueId=_target_id ,
    childLinkIndex=-1,                 # -1 means the base link of the child
    jointType=p.JOINT_FIXED,
    jointAxis=[0, 0, 0],
    parentFramePosition=[0, 0, 0.02],   # Position of zone relative to workspace
    childFramePosition=[0, 0, 0],      # Position of zone relative to its own origin
)
p.resetDebugVisualizerCamera(
    cameraDistance=0.9,
    cameraYaw=90,
    cameraPitch=-40,
    cameraTargetPosition=[0, 0, 0.1],
)

# Re-enable rendering.
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

for _ in range(100):
    p.stepSimulation()
    time.sleep(0.01)

""" Camera methods:
"""


# image_size = np.array([320, 240])

# visuals_mode = "default"
# X_MIN_REAL = 0.15
# X_MAX_REAL = 0.6
# Y_MIN_REAL = -0.3048
# Y_MAX_REAL = 0.3048
# DEFAULT_CAMERA_POSE = (1.1, 0, 0.75)
# DEFAULT_CAMERA_ORIENTATION = (np.pi / 4, np.pi, -np.pi / 2)
# IMAGE_WIDTH = 320
# IMAGE_HEIGHT = 240
# CAMERA_INTRINSICS = (0.803 * IMAGE_WIDTH, 0,IMAGE_WIDTH / 2.0,0,0.803 * IMAGE_WIDTH,IMAGE_HEIGHT / 2.0,0,0,1,)
# WORKSPACE_BOUNDS = np.array(((0.15, -0.5), (0.7, 0.5)))
# WORKSPACE_BOUNDS_REAL = np.array(((X_MIN_REAL, Y_MIN_REAL), (X_MAX_REAL, Y_MAX_REAL)))
# WORKSPACE_URDF_PATH_REAL = "third_party/py/ibc/environments/assets/workspace_real.urdf"
# IMAGE_WIDTH_REAL = 320
# IMAGE_HEIGHT_REAL = 180
# CAMERA_POSE_REAL = (0.75, 0, 0.5)
# CAMERA_ORIENTATION_REAL = (np.pi / 5, np.pi, -np.pi / 2)
# CAMERA_INTRINSICS_REAL = (0.803 * IMAGE_WIDTH_REAL,0,IMAGE_WIDTH_REAL / 2.0,0,0.803 * IMAGE_WIDTH_REAL,IMAGE_HEIGHT_REAL / 2.0,0,0,1,)
# # Mimic RealSense D415 camera parameters.
# if visuals_mode == "default":
#     _camera_pose = DEFAULT_CAMERA_POSE
#     _camera_orientation = DEFAULT_CAMERA_ORIENTATION
#     workspace_bounds = WORKSPACE_BOUNDS
#     _camera_instrinsics = CAMERA_INTRINSICS
#     _workspace_urdf_path = WORKSPACE_URDF_PATH
# else:
#     _camera_pose = CAMERA_POSE_REAL
#     _camera_orientation = CAMERA_ORIENTATION_REAL
#     workspace_bounds = WORKSPACE_BOUNDS_REAL
#     _camera_instrinsics = CAMERA_INTRINSICS_REAL
#     _workspace_urdf_path = WORKSPACE_URDF_PATH_REAL


# def calc_camera_params(image_size):
#     intrinsics = _camera_instrinsics

#     # Set default camera poses.
#     front_position = _camera_pose
#     front_rotation = _camera_orientation
#     front_rotation = p.getQuaternionFromEuler(front_rotation)
#     # Default camera configs.
#     zrange = (0.01, 10.0)

#     # OpenGL camera settings.
#     lookdir = np.float32([0, 0, 1]).reshape(3, 1)
#     updir = np.float32([0, -1, 0]).reshape(3, 1)
#     rotation = p.getMatrixFromQuaternion(front_rotation)
#     rotm = np.float32(rotation).reshape(3, 3)
#     lookdir = (rotm @ lookdir).reshape(-1)
#     updir = (rotm @ updir).reshape(-1)
#     lookat = front_position + lookdir
#     focal_len = intrinsics[0]
#     znear, zfar = zrange
#     viewm = p.computeViewMatrix(front_position, lookat, updir)
#     fovh = (image_size[0] / 2) / focal_len
#     fovh = 180 * np.arctan(fovh) * 2 / np.pi

#     # Notes: 1) FOV is vertical FOV 2) aspect must be float
#     aspect_ratio = image_size[1] / image_size[0]
#     projm = p.computeProjectionMatrixFOV(
#         fovh, aspect_ratio, znear, zfar
#     )


#     return viewm, projm, front_position, lookat, updir



# # def _render_camera(self, image_size):

#     """Render RGB image with RealSense configuration."""
# viewm, projm, _, _, _ = calc_camera_params(image_size)
# _, _, color, _, _ = p.getCameraImage(
#     width=image_size[1],
#     height=image_size[0],
#     viewMatrix=viewm,
#     projectionMatrix=projm,
#     flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
#     renderer=p.ER_BULLET_HARDWARE_OPENGL,
# )


# # Get color image.
# color_image_size = (image_size[0], image_size[1], 4)
# color = np.array(color, dtype=np.uint8).reshape(color_image_size)
# color = color[:, :, :3]  # remove alpha channel

# image =  color.astype(np.uint8)   
# plt.imshow(image)
# plt.title("Simulated Camera View")
# plt.axis("off")
# plt.show()


# pi2 = math.pi * 2

# obs_dict = collections.OrderedDict(
#     block_translation=spaces.Box(low=-5, high=5, shape=(3,)),  # x,y
#     block_orientation=spaces.Box(low=-pi2, high=pi2, shape=(1,)),  # phi
#     effector_translation=spaces.Box(
#         low=workspace_bounds[0] - 0.1,  # Small buffer for to IK noise.
#         high=workspace_bounds[1] + 0.1,
#     ),  # x,y
#     effector_target_translation=spaces.Box(
#         low=workspace_bounds[0] - 0.1,  # Small buffer for to IK noise.
#         high=workspace_bounds[1] + 0.1,
#     ),  # x,y
#     target_translation=spaces.Box(low=-5, high=5, shape=(3,)),  # x,y
#     target_orientation=spaces.Box(
#         low=-pi2,
#         high=pi2,
#         shape=(1,),
#     ),  # theta
# )
# if image_size is not None:
#     obs_dict["rgb"] = spaces.Box(
#         low=0, high=255, shape=(image_size[0], image_size[1], 3), dtype=np.uint8
#     )
# obs_dic =  spaces.Dict(obs_dict)
# print(obs_dic)


# from PIL import Image
# import os

# def render_n_images(robot_id, block_id, max_img_number):
#     os.makedirs("rendered_images", exist_ok=True)

#     num_joints = p.getNumJoints(robot_id)

#     for i in range(max_img_number):
#         # Set a random robot joint configuration
#         for j in range(num_joints):
#             random_angle = np.random.uniform(-1.0, 1.0)
#             p.resetJointState(robot_id, j, random_angle)

#         # Randomize block position in workspace
#         rand_x = np.random.uniform(0.3, 0.7)
#         rand_y = np.random.uniform(-0.2, 0.2)
#         p.resetBasePositionAndOrientation(block_id, [rand_x, rand_y, 0.02], [0, 0, 0, 1])

#         # Step physics
#         for _ in range(5):
#             p.stepSimulation()
#             time.sleep(0.01)

#         # Render image
#         viewm, projm, _, _, _ = calc_camera_params(image_size)
#         _, _, color, _, _ = p.getCameraImage(
#             width=image_size[1],
#             height=image_size[0],
#             viewMatrix=viewm,
#             projectionMatrix=projm,
#             renderer=p.ER_BULLET_HARDWARE_OPENGL
#         )
#         color = np.array(color, dtype=np.uint8).reshape((image_size[0], image_size[1], 4))[:, :, :3]

#         # Save image
#         im = Image.fromarray(color)
#         im.save(f"rendered_images/image_{i:04d}.png")

#         if i % 100 == 0:
#             print(f"Rendered {i} images...")

#     print("✅ Finished rendering images.")

# render_n_images(_robot.gripperarm, _block_id, 100)


""" compute state
"""
block_position = p.getBasePositionAndOrientation(_block_id)[0]
block_orientation = p.getBasePositionAndOrientation(_block_id)[1]
block_pose = Pose3d(
    rotation=transform.Rotation.from_quat(block_orientation),
    translation=block_position,)
def _yaw_from_pose(pose):
    return np.array([pose.rotation.as_euler("xyz", degrees=False)[-1]])

robot_pose = _robot.forward_kinematics()
obs = collections.OrderedDict(
    block_translation=block_pose.translation[0:2],
    block_orientation=_yaw_from_pose(block_pose),
    gripper_translation_left=robot_pose.translation_left[0:2],
    gripper_translation_right=robot_pose.translation_right[0:2],
    effector_target_translation=_target_effector_pose.translation[0:2],
    target_translation=_target_pose.translation[0:2],
    target_orientation=_yaw_from_pose(self._target_pose),
)
if self._image_size is not None:
    obs["rgb"] = self._render_camera(self._image_size)