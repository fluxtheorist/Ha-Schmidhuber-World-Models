import gymnasium as gym
import numpy as np
from PIL import Image
import os

# setting up box 2D racer envriornment
env = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=True)

# Reset environment and get first observation
obs, info = env.reset()

# Print observation shape and action space
print(f"Observation shape: {obs.shape}")
print(f"Action space: {env.action_space}")
print(f"Action space type: {type(env.action_space)}")

# Create outputs directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# Run ~100 steps and save 3-4 frames
frames_to_save = [10, 30, 60, 90]  # Steps at which to save frames
num_steps = 100

for step in range(num_steps):
    # Sample random action from action space
    action = env.action_space.sample()

    # Take a step in the environment
    obs, reward, terminated, truncated, info = env.step(action)

    # Save frame if this is one of the selected steps
    if step in frames_to_save:
        frame = obs  # obs is the rgb_array
        img = Image.fromarray(frame.astype(np.uint8))
        img.save(f"outputs/frame_step_{step}.png")
        print(f"Saved frame at step {step}")

    # Reset if episode ends
    if terminated or truncated:
        print(f"Episode ended at step {step}")
        obs, info = env.reset()

env.close()
print(f"\nFinal observation shape: {obs.shape}")
