import gymnasium as gym
import numpy as np
from PIL import Image
import os

# Settings
NUM_EPISODES = 50
MAX_STEPS = 500
os.makedirs("outputs", exist_ok=True)

# Storage for all frames
all_frames = []
all_actions = []
episode_lengths = []
all_rewards = []
# Create enviornment
env = gym.make("CarRacing-v3", continuous=True)

for episode in range(NUM_EPISODES):
    obs, _ = env.reset()
    episode_frames = 0

    for step in range(MAX_STEPS):
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
episode_lenghts = np.array(episode_lengths)
all_rewards = np.array(all_rewards)


print(f"Collected {len(all_frames)} frames with shape {all_frames.shape}")
print(f"Collected {len(all_actions)} actions with shape {all_actions.shape}")
print(f"Collected {len(episode_lengths)} episodes")
print(f"Rewards: {all_rewards.shape}")
print(f"Reward range: {all_rewards.min():.2f} to {all_rewards.max():.2f}")


np.save("outputs/frames.npy", all_frames)
np.save("outputs/actions.npy", all_actions)
np.save("outputs/episode_lengths.npy", episode_lengths)
np.save("outputs/rewards.npy", all_rewards)
