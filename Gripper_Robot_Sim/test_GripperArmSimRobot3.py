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
robot = GripperArmSimRobot(physics_client)

# Load block
block_start_pos = [0.4, -0.3, 0.04]  # height must match block URDF base
block_start_ori = p.getQuaternionFromEuler([0, 0, 0])
block_id = p.loadURDF("cube_small.urdf", block_start_pos, block_start_ori)

# Let everything settle
for _ in range(100):
    p.stepSimulation()
    time.sleep(1 / 240.)

# Pick and place
place_position = np.array([0.4, 0.3, 0.04])  # place on other side

robot.set_target_pick_n_place_the_block(place_position, np.array(block_start_pos))

# Let simulation run for a while
for _ in range(300):
    p.stepSimulation()
    time.sleep(1 / 240.)

p.disconnect()
