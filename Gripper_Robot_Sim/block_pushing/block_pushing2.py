import gym
from gym import spaces
import numpy as np

class MyCustomEnv(gym.Env):

    def __init__(self):
        super(MyCustomEnv, self).__init__()

        # Define action and observation spaces
        # Example: 2D continuous action, 3D continuous observation
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

        # Environment state
        self.state = None
        self.goal = np.array([0.0, 0.0, 0.0])
        self.step_count = 0
        self.max_steps = 100

    def reset(self):
        """Resets the environment to an initial state and returns an observation."""
        self.state = np.random.uniform(low=-1.0, high=1.0, size=(3,))
        self.step_count = 0
        return self.state

    def step(self, action):
        """Executes one step in the environment given an action."""
        self.step_count += 1

        # Apply action (dummy dynamics)
        pdb.set_trace()
        self.state = self.state + 0.1 * np.tanh(action)

        # Calculate reward (e.g., distance to goal)
        reward = -np.linalg.norm(self.state - self.goal)

        # Check if done
        done = self.step_count >= self.max_steps or np.linalg.norm(self.state - self.goal) < 0.1

        # Optional info dict
        info = {}

        return self.state, reward, done, info

    def render(self, mode='human'):
        """Renders the environment (optional)."""
        print(f"State: {self.state}")

    def close(self):
        """Performs cleanup (if necessary)."""
        pass













env = MyCustomEnv()

# Reset the environment
obs = env.reset()
print("Initial observation:", obs)

# Run for a few steps
for step in range(10):
    action = env.action_space.sample()  # Take a random action
    obs, reward, done, info = env.step(action)
    print(f"Step {step + 1}:")
    print(f"  Action: {action}")
    print(f"  Observation: {obs}")
    print(f"  Reward: {reward}")
    print(f"  Done: {done}")
    if done:
        print("Episode finished!")
        break

# Clean up
env.close()