import numpy as np
from stable_baselines3 import DQN
from src.env.intrusion_env import FirewallEnv
from src.utils import helpers
import time

def train_agent():
    """
    Loads processed data, initializes the environment, and trains the DQN agent.
    (From Cells 19 & 20)
    """
    print("Loading processed training data...")
    try:
        X_train_scaled = np.load(helpers.X_TRAIN_PATH)
        y_train = np.load(helpers.Y_TRAIN_PATH)
    except FileNotFoundError:
        print("Error: Processed data not found. Please run 'src/features/build_features.py' first.")
        return

    print("Initializing FirewallEnv...")
    env = FirewallEnv(X_train_scaled, y_train)

    # Initialize model (Cell 19)
    # Note: We don't need the Monitor wrapper here, stable-baselines handles it.
    print("Initializing DQN model...")
    model = DQN(
        "MlpPolicy", 
        env, 
        verbose=1, 
        buffer_size=50000, 
        learning_starts=1000
    )

    print("Starting model training (50,000 timesteps)...")
    start_time = time.time()
    model.learn(total_timesteps=50000)
    end_time = time.time()
    
    print(f"\nTraining complete in {end_time - start_time:.2f} seconds.")

    # Save model (Cell 20)
    model.save(helpers.MODEL_PATH)
    print(f"Model saved to {helpers.MODEL_PATH}")

if __name__ == "__main__":
    helpers.create_dirs() # Ensure directories exist
    train_agent()