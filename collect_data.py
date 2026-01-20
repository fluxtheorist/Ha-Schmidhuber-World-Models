import gymnasium as gym
import numpy as np
from PIL import Image
import os

# Settings
NUM_EPISODES = 5
MAX_STEPS = 200
os.makedirs("outputs", exist_ok=True)

# Storage for all frames
all_frames = []

# Create enviornment
env = gym.make("CarRacing-v3", continuous=True)

for episode in range(NUM_EPISODES):
    obs, _ = env.reset()

    for step in range(MAX_STEPS):
        img = Image.fromarray(obs).resize((64, 64))
        all_frames.append(np.array(img))
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

env.close()

# Convert to np array and save
all_frames = np.array(all_frames)
print(f"Collected {len(all_frames)} frames with shape {all_frames.shape}")
np.save("outputs/frames.npy", all_frames)
