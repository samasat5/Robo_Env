""" 
testing the pick and place function
"""

import pybullet as p
import pybullet_data
import time
import numpy as np

from block_pushing.utils import utils_pybullet
from block_pushing.utils.franka_panda_sim_robot import GripperArmSimRobot 
from block_pushing.utils.pose3d_gripper import Pose3d_gripper  

# Connect to simulation
physics_client = p.connect(p.GUI)  # Use GUI for visualization
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setTimeStep(1. / 240.)
p.resetDebugVisualizerCamera(
    cameraDistance=1.5,
    cameraYaw=90,
    cameraPitch=-40,
    cameraTargetPosition=[0, 0, 0.1],
)

# Load plane
plane_id = p.loadURDF("plane.urdf")

# Create robot
INITIAL_JOINT_POSITIONS = np.array(
    [
        0.0, 
        -0.5235987755982988, 
        0.0, -1.0471975511965976, 
        0.0, 1.5707963267948966, 
        0.0, 
        0.0, 
        0.0])
BLOCK_URDF_PATH = "third_party/py/envs/assets/block2.urdf"
PLANE_URDF_PATH = "third_party/bullet/examples/pybullet/gym/pybullet_data/" "plane.urdf"
WORKSPACE_URDF_PATH = "third_party/py/envs/assets/workspace.urdf"
ZONE_URDF_PATH = "third_party/py/envs/assets/zone.urdf"
INSERT_URDF_PATH = "third_party/py/envs/assets/insert.urdf"
workspace_uid = utils_pybullet.load_urdf(
    p,
    WORKSPACE_URDF_PATH,
    basePosition=[0.35, 0, 0.0],)

robot = GripperArmSimRobot(p,INITIAL_JOINT_POSITIONS)
target_id = utils_pybullet.load_urdf(p,ZONE_URDF_PATH,
                                      [0.4999, -0.36, 0], 
                                      useFixedBase=True)
block_id = utils_pybullet.load_urdf(p, BLOCK_URDF_PATH, 
                                     [0.2, 0.47, 0.01],
                                     useFixedBase=False)


# Load block

# Pick and place 
# place_position2 = np.array([0.4999, -0.36, 0.1]) 
place_position = np.array([0.4999, -0.36, 0]) # place on other side
# place_position = np.array([0.35, 0, 0.15])
block_position = np.array([0.2, 0.47, 0.01])
opening_width =0.04+0.0001
closing_width = -0.005
# robot.set_target_pick_the_block(block_position)
# robot.set_target_pick_n_place_the_block (place_position, block_position)
force = 2
size_of_the_block = 0.04
opening_width = size_of_the_block + 0.0001 # grabbing size to grasp the block
robot.set_the_fingers_open_close(opening_width,force)
for _ in range(50):
    p.stepSimulation()
    time.sleep(1 / 240.0)
time.sleep(2)
force = 7
feasible_block_position = block_position + np.array([0, 0, 0.01])
robot.move_gripper_to_target(block_position,force)
for _ in range(100):
    p.stepSimulation()
    time.sleep(1 / 240.0)
closing_width = -0.005
force = 1
robot.set_the_fingers_open_close(closing_width,force)
for _ in range(100):
    p.stepSimulation()
    time.sleep(1 / 240.0)
time.sleep(2)
force = 0.2
feasible_place_position = place_position + np.array([0, 0, 0.1])
robot.move_gripper_to_target(feasible_place_position, force)
for _ in range(500):
    p.stepSimulation()
    time.sleep(1 / 150)

p.disconnect()
