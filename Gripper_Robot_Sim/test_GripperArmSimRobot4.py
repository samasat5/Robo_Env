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

# dataset = 

for i in range(10):
    action = env.action_space.sample()  # random action
    # action = dataset[i]['action'].squeeze(0).numpy()
    obs, reward, done, info = env.step(action)

    print(f"Step {step}: reward={reward:.3f}, done={done}")
    print("Event info:", info)

    # Optional: render as RGB
    img = env.render(mode="rgb_array")
    plt.imshow(img)
    plt.title(f"Step {step}")
    plt.pause(0.1)
    step += 1

env.close()