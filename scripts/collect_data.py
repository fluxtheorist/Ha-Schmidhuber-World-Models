import sys

sys.path.append("..")

import gymnasium as gym
import numpy as np
from PIL import Image
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--iter", type=int, default=0, help="Iteration number")
parser.add_argument("--episodes", type=int, default=50, help="Number of episodes")
parser.add_argument("--max_steps", type=int, default=500, help="Max steps per episode")
args = parser.parse_args()

# Settings
output_dir = f"../outputs/iter{args.iter}"
os.makedirs(output_dir, exist_ok=True)

# Storage for all frames
all_frames = []
all_actions = []
episode_lengths = []
all_rewards = []

# Create environment
env = gym.make("CarRacing-v3", continuous=True)

for episode in range(args.episodes):
    obs, _ = env.reset()
    episode_frames = 0

    for step in range(args.max_steps):
        img = Image.fromarray(obs).resize((64, 64))
        all_frames.append(np.array(img))
        action = env.action_space.sample()
        all_actions.append(action)
        episode_frames += 1

        obs, reward, terminated, truncated, info = env.step(action)
        all_rewards.append(reward)

        if terminated or truncated:
            break
    episode_lengths.append(episode_frames)
    print(f"Episode {episode}: {episode_frames} frames")

env.close()

# Convert to np array and save
all_frames = np.array(all_frames)
all_actions = np.array(all_actions)
episode_lengths = np.array(episode_lengths)
all_rewards = np.array(all_rewards)

print(f"Collected {len(all_frames)} frames with shape {all_frames.shape}")
print(f"Collected {len(all_actions)} actions with shape {all_actions.shape}")
print(f"Collected {len(episode_lengths)} episodes")
print(f"Rewards: {all_rewards.shape}")
print(f"Reward range: {all_rewards.min():.2f} to {all_rewards.max():.2f}")

np.save(f"{output_dir}/frames.npy", all_frames)
np.save(f"{output_dir}/actions.npy", all_actions)
np.save(f"{output_dir}/episode_lengths.npy", episode_lengths)
np.save(f"{output_dir}/rewards.npy", all_rewards)
print(f"Saved to {output_dir}/")
