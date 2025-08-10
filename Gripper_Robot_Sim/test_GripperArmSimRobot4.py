import gym
import time
import numpy as np
import pybullet as p
import gym
import matplotlib.pyplot as plt
from block_picking.utils.franka_panda_sim_robot import GripperArmSimRobot
from block_picking.block_picking3 import BlockPick
from block_picking.utils.franka_panda_sim_robot import GripperArmSimRobot
import block_picking.block_picking3
from block_picking.utils.franka_panda_sim_robot import GripperArmSimRobot 

# import pybullet_data
# import math
# INITIAL_JOINT_POSITIONS = np.array(
#     [
#         0.0, 
#         -0.5235987755982988, 
#         0.0, -1.0471975511965976, 
#         0.0, 1.5707963267948966, 
#         0.0, 
#         0.0, 
#         0.0])
# physics_client = p.connect(p.GUI)  # Use GUI for visualization
# p.setAdditionalSearchPath(pybullet_data.getDataPath())
# p.setGravity(0, 0, -9.81)
# p.setTimeStep(1. / 240.)
# p.resetDebugVisualizerCamera(
#     cameraDistance=1.3,
#     cameraYaw=100,
#     cameraPitch=-30,
#     cameraTargetPosition=[0, 0, 0.1],
# )
# # Load plane
# plane_id = p.loadURDF("plane.urdf")
# robot =  p.loadURDF("franka_panda/panda.urdf",[0,0,0], [0,0,0,1], useFixedBase = True ) # each urdf is basivally a set of links . # fix base for objects so it doesnt move
# time.sleep(2)

# movable_joints = []
# for i in range(p.getNumJoints(robot)):
#     joint_info = p.getJointInfo(robot, i)
#     joint_type = joint_info[2]
#     joint_name = joint_info[1].decode("utf-8")
#     if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]: # only the joints that move 
#         movable_joints.append(i)
#         p.addUserDebugText(
#             f"{i}: {joint_name}",
#             [0, 0, 0.1],
#             parentObjectUniqueId=robot,
#             parentLinkIndex=i,
#             textColorRGB=[1, 0, 0],
#             textSize=1.0,)

# # move the joints one by one
# for joint_index in movable_joints:
#     print(f"Moving joint {joint_index}")
#     for t in range(240):  
#         angle = 0.5 * math.sin(t*0.05) # using sin to have a oscillating movement 
#         p.setJointMotorControl2(
#             robot,
#             joint_index,
#             controlMode=p.POSITION_CONTROL,
#             targetPosition=angle,
#             force=100,)
#         p.stepSimulation()
#         time.sleep(1 / 240)


# while True:
#     p.stepSimulation()
#     time.sleep(1)

# env = gym.make("BlockPick-v0")
env = BlockPick(image_size=None)

# obs = env.reset(seed=42)
obs = env.reset()
done = False
step = 0

trajectory = []
z_height = env.effector_height 

block_pos = obs['block_translation']
target_pos = obs['target_translation']
actions = [
    np.r_[block_pos[:2], env.effector_height],  
    np.r_[target_pos[:2], env.effector_height],  
]


for i, action in enumerate(actions):
    # action = env.action_space.sample()  # random action
    # action = dataset[i]['action'].squeeze(0).numpy()
    obs, reward, done, info = env.step(action)

    frame = env.render() if hasattr(env, 'render') else None

    trajectory.append({
        "step": i,
        "obs": obs,
        "action": action.tolist(),
        "reward": reward,
        "done": done,
        "info": info,
        "frame": frame  # optional, remove if not needed
    })

env.close()