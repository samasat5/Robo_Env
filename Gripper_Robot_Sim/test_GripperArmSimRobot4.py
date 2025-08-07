import gym
import time
import numpy as np

import gym
import matplotlib.pyplot as plt
from block_pushing.utils.franka_panda_sim_robot import GripperArmSimRobot
from block_pushing.block_picking3 import BlockPick
import block_pushing.block_picking3


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