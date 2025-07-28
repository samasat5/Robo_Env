""" 
Testing XArmRobotSim : an XArm with different end-effectors like  "suction", "cylinder", "cylinder_real"
(the Authors)

"""


import pybullet as p
import pybullet_data
import time
import numpy as np
from scipy.spatial.transform import Rotation
from block_pushing.utils.xarm_sim_robot import XArmSimRobot
from block_pushing.utils.pose3d import Pose3d

#  Start PyBullet
client = p.connect(p.GUI)
p.setGravity(0, 0, -9.81)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")

block = p.loadURDF("block2.urdf",[0.2, 0.5, 0.01]  , useFixedBase = False )

p.resetDebugVisualizerCamera(
    cameraDistance=1.5,
    cameraYaw=100,
    cameraPitch=-30,
    cameraTargetPosition=[0, 0, 0.1],
)

# Create robot (optionally with end effector: "suction", "cylinder", "cylinder_real")
robot = XArmSimRobot(pybullet_client=p, end_effector="suction")

# Reset joints to home position
print("\n[TEST] reset_joints")
robot.reset_joints(robot.initial_joint_positions)
for _ in range(100):
    p.stepSimulation()
    time.sleep(1 / 240.0)
time.sleep(1)

# Get joint positions, velocities, torques
print("\n[TEST] get_joints_measured")
positions, velocities, torques = robot.get_joints_measured()
print("Positions:", positions)
print("Velocities:", velocities)
print("Torques:", torques)

#  Get joint positions separately
print("\n[TEST] get_joint_positions")
print("Joint Positions:", robot.get_joint_positions())

# Test forward kinematics
print("\n[TEST] forward_kinematics")
pose = robot.forward_kinematics()
print("FK pose translation:", pose.translation)
print("FK pose rotation (quat):", pose.rotation.as_quat())

# Test Inverse kinematics with a small upward shift
print("\n[TEST] inverse_kinematics")
new_pose = Pose3d(
    # translation=pose.translation + np.array([0, 0, 0.05]), 
    translation=np.array([0.2, 0.5, 0.2] ), 
    rotation=pose.rotation)
ik_solution = robot.inverse_kinematics(new_pose)
print("IK Joint Angles (target_joint_positions):", ik_solution)

print("\n[DEBUG] End-effector position before IK move:")
print("Original translation:", pose.translation)
print("Expected new center:", new_pose.translation)

# Apply the pose via IK
print("\n[TEST] set_target_effector_pose")
robot.set_target_effector_pose(new_pose)
for _ in range(100):
    p.stepSimulation()
    time.sleep(1 / 240.0)
time.sleep(1)

# Apply joint position control directly
print("\n[TEST] set_target_joint_positions")
offset_positions = robot.get_joint_positions() + np.deg2rad([10] * len(robot._joint_indices))
robot.set_target_joint_positions(offset_positions)
for _ in range(100):
    p.stepSimulation()
    time.sleep(1 / 240.0)

# Apply joint velocity control
print("\n[TEST] set_target_joint_velocities")
robot.set_target_joint_velocities([0.0] * len(robot._joint_indices))
for _ in range(100):
    p.stepSimulation()
    time.sleep(1 / 240.0)

# Make robot semi-transparent
print("\n[TEST] set_alpha_transparency")
robot.set_alpha_transparency(0.3)







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