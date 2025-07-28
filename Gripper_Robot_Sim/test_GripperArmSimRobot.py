""" 
Testing GripperArmSimRobot : an Franka Panda robot with a gripper (two fingers)
(Mine)


"""


import pybullet as p
import pybullet_data
import time
import numpy as np
import pdb

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



client = p.connect(p.GUI)
p.setGravity(0, 0, -9.81)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")

p.resetDebugVisualizerCamera(
    cameraDistance=1,
    cameraYaw=100,
    cameraPitch=-30,
    cameraTargetPosition=[0, 0, 0.1],
)

block = p.loadURDF("block2.urdf",[0.2, 0.5, 0.01]  , useFixedBase = False )

# Create robot with end effector
robot = GripperArmSimRobot(pybullet_client=p)

# Reset joints to home position
print("\n[TEST] reset_joints")
robot.reset_joints(robot.initial_joint_positions)

# Step simulation
for _ in range(100):
    p.stepSimulation()
    time.sleep(1 / 240.0)
time.sleep(2)



# Get joint positions, velocities, torques
print("\n[TEST] get_joints_measured")
positions, velocities, torques = robot.get_joints_measured()
print("Positions:", positions)
print("Velocities:", velocities)
print("Torques:", torques)

# Get joint positions separately
print("\n[TEST] get_joint_positions")
print("Joint Positions:", robot.get_joint_positions())

# Test FK
print("\n[TEST] forward_kinematics")## uses getLinkState
pose = robot.forward_kinematics()  # computes the current 3D pose (position + orientation) of the robot's end effector in  
print("FK pose translation left:", pose.translation_left) # x,y,z
print("FK pose rotation left (quat):", pose.rotation_left.as_quat()) #quaternion [x, y, z, w]

# test Inverse Kinematics
print("\n[TEST] inverse_kinematics")
# new_translation_left = pose.translation_left + np.array([0, 0, -1]) 
# new_translation_right = pose.translation_right + np.array([0, 0, -1])  
target_center = np.array([0.2, 0.5, 0.01 + 0.015])
offset = np.array([0.03, 0, 0])  # assume fingers are 6cm apart
new_translation_left = target_center - offset
new_translation_right = target_center + offset
 
new_pose = Pose3d_gripper(translation_left=new_translation_left,
                          translation_right=new_translation_right,
                          rotation_left=pose.rotation_left, 
                          rotation_right=pose.rotation_left) #Create a new Pose3d with same orientation but new position
ik_solution = robot.inverse_kinematics(new_pose) #Solve Inverse Kinematics to find joint angles to reach new_pose
print("IK Joint Angles (target_joint_positions):", ik_solution)





# Apply pose via IK (going toward the block and grasping it)
print("\n[TEST] set_target_effector_pose")
size_of_the_block = 0.04
opening_width = size_of_the_block + 0.0001 # grabbing size to grasp the block
robot.set_the_fingers_open_close(opening_width)
for _ in range(50):
    p.stepSimulation()
    time.sleep(1 / 240.0)
force = 5
robot.set_target_effector_pose(new_pose,force)
for _ in range(100):
    p.stepSimulation()
    time.sleep(1 / 240.0)
closing_width = - 0.0001
robot.set_the_fingers_open_close(closing_width)
for _ in range(50):
    p.stepSimulation()
    time.sleep(1 / 240.0)
    
    
    
# moving the block to another place
target_center = np.array([0.4, -0.3, 0.01 + 0.15])
offset = np.array([0.03, 0, 0])  # assume fingers are 6cm apart
new_translation_left = target_center - offset
new_translation_right = target_center + offset
new_pose = Pose3d_gripper(translation_left=new_translation_left,
                          translation_right=new_translation_right,
                          rotation_left=pose.rotation_left, 
                          rotation_right=pose.rotation_left) #Create a new Pose3d with same orientation but new position
ik_solution = robot.inverse_kinematics(new_pose)
force = 1 #lowering the speed to prevent the block from falling
robot.set_target_effector_pose(new_pose,force)
for _ in range(100):
    p.stepSimulation()
    time.sleep(1 / 240.0)
    

    
print("___________________testing if the location of end effector==the point it want to go_____________")
left_finger_pos, _ = p.getLinkState(robot.gripperarm, robot.left_finger)[:2]
right_finger_pos, _ = p.getLinkState(robot.gripperarm, robot.right_finger)[:2]
print("\n[RESULT] Finger locations after IK:")
print("Left finger position after the block reachout :", np.round(left_finger_pos, 4))
print("Right finger position after the block reachout :", np.round(right_finger_pos, 4))
print("block location :",[0.2, 0.5, 0.01] )
center = (pose.translation_left + pose.translation_left) /2
print(pose.translation_left)
# print(pose.translation_right)
# print(center)
position = robot.forward_kinematics().translation_left
print("End-effector position:", position)




# # Apply joint position control directly
# print("\n[TEST] set_target_joint_positions")
# offset_positions = robot.get_joint_positions() + np.deg2rad([90]*9)  # gives initial joint (what joints? joint_indices ! which are rev and peris joints so not all the joints) angles +...  # [5, 5, 5, 5,.., 5, 5] degrees → radians:
# # offset_positions = robot.initial_joint_positions + np.deg2rad([0, 90, 0, 0, 0, 0,0,0,0])
# robot.set_target_joint_positions(offset_positions)
# for _ in range(100):
#     p.stepSimulation()
#     time.sleep(1 / 240.0)

# # Apply joint velocity control
# print("\n[TEST] set_target_joint_velocities")
# zero_velocity = np.zeros(9) # array([0., 0., 0., 0., 0., 0.])   # we need to stop the movement becasue after the rotation order, it'll continue rotating forever (like a motor with no brake) — unless you later call:
# robot.set_target_joint_velocities(zero_velocity)
# for _ in range(100):
#     p.stepSimulation()
#     time.sleep(1 / 240.0)

time.sleep(5)

# # 12. Set transparency
# print("\n[TEST] set_alpha_transparency")
# robot.set_alpha_transparency(0.5)
# for _ in range(100):
#     p.stepSimulation()
#     time.sleep(1 / 240.0)




# # Done
# print("\n[TEST COMPLETED]")
# p.disconnect()
















# import gym
# import time
# import numpy as np

# import gym
# import matplotlib.pyplot as plt

# env = gym.make("BlockInsert-v0")

# obs = env.reset()
# done = False
# step = 0

# for i in range(10):
#     action = env.action_space.sample()  # random action
#     # action = dataset[i]['action'].squeeze(0).numpy()
#     obs, reward, done, info = env.step(action)

#     print(f"Step {step}: reward={reward:.3f}, done={done}")
#     print("Event info:", info)

#     # Optional: render as RGB
#     img = env.render(mode="rgb_array")
#     plt.imshow(img)
#     plt.title(f"Step {step}")
#     plt.pause(0.1)
#     step += 1

# env.close()