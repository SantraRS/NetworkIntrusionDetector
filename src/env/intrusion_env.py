import gymnasium as gym
from gymnasium import spaces
import numpy as np

class FirewallEnv(gym.Env):
    """
    Custom Gym environment for the network intrusion detection agent.
    (From Cell 17)
    """
    def __init__(self, X_data, y_data):
        super(FirewallEnv, self).__init__()
        self.X_data = X_data
        self.y_data = y_data
        self.current_step = 0
        self.max_steps = len(X_data)

        # Action space: 0 (ALLOW), 1 (DENY)
        self.action_space = spaces.Discrete(2)
        
        # Observation space: matches the features
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(X_data.shape[1],), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        obs = self.X_data[self.current_step]
        info = {} # Gymnasium standard
        return obs, info

    def step(self, action):
        true_label = self.y_data[self.current_step]
        reward = 0

        if action == 0 and true_label == 0:   # allow benign
            reward = 1
        elif action == 0 and true_label == 1: # allow attack
            reward = -10
        elif action == 1 and true_label == 0: # deny benign
            reward = -5
        elif action == 1 and true_label == 1: # deny attack
            reward = 10

        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False # We don't have a time limit separate from episode end

        if not terminated:
            obs = self.X_data[self.current_step]
        else:
            # Return a zero observation if terminated
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        
        info = {} # Gymnasium standard

        return obs, reward, terminated, truncated, info

print("FirewallEnv class loaded.")