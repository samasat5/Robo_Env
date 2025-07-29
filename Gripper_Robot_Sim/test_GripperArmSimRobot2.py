
import pybullet as p
import pybullet_data
import time
import numpy as np
import pdb
from scipy.spatial import transform

from block_pushing.utils.xarm_sim_robot import XArmSimRobot
from block_pushing.utils.franka_panda_sim_robot import GripperArmSimRobot

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
    

# print("robot current state",robot._get_current_translation_orientation())

    
# time.sleep(3)



import pybullet
from block_pushing.block_pushing2 import BlockPick  # adjust import path
import numpy as np

# 1. Instantiate the environment
env = BlockPick(
    control_frequency=10.0,
    image_size=(128, 128),
    shared_memory=False,
    seed=42,
    goal_dist_tolerance=0.01,
    effector_height=0.07,  # try a safe height > 0
    visuals_mode="default",
    abs_action=False
)

# 2. Reset the environment
try:
    obs = env.reset()
    print("✅ Reset completed successfully.")
except Exception as e:
    print("❌ Reset failed:", e)

# 3. Inspect the initial observation
print("Initial observation:", obs)
print("Observation shape:", np.shape(obs))

env.close()

    