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
BLOCK_URDF_PATH = "third_party/py/envs/assets/block.urdf"
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
                                      [0.4999, -0.36, 0.1], 
                                      useFixedBase=True)
block_id = utils_pybullet.load_urdf(p, BLOCK_URDF_PATH, 
                                     [0.2, 0.47, 0.01],
                                     useFixedBase=False)


# Load block

# Let everything settle
for _ in range(100):
    p.stepSimulation()
    time.sleep(1 / 240.)

# Pick and place
place_position = np.array([0.4999, -0.36, 0.1])  # place on other side
block_position = np.array([0.2, 0.47, 0.01])
robot.set_target_pick_n_place_the_block(place_position, block_position)

# Let simulation run for a while
for _ in range(300):
    p.stepSimulation()
    time.sleep(1 / 240.)

p.disconnect()
