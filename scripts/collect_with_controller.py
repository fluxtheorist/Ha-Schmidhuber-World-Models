"""
Collect data using trained controller.
Step 1 of iterative training.
"""

import sys

sys.path.append("..")

import torch
import numpy as np
import gymnasium as gym
from PIL import Image
import os
import argparse
from models.vae import ConvVAE
from models.mdn_rnn import MDNRNN
from models.controller import Controller

parser = argparse.ArgumentParser()
parser.add_argument(
    "--iter",
    type=int,
    required=True,
    help="Iteration to collect FOR (loads models from iter-1)",
)
parser.add_argument("--episodes", type=int, default=50, help="Number of episodes")
parser.add_argument("--max_steps", type=int, default=500, help="Max steps per episode")
args = parser.parse_args()

# Load from previous iteration, save to current
load_dir = f"../outputs/iter{args.iter - 1}"
output_dir = f"../outputs/iter{args.iter}"
os.makedirs(output_dir, exist_ok=True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print(f"Loading models from {load_dir}")
print(f"Saving data to {output_dir}")

# Load models
vae = ConvVAE(latent_dim=32)
vae.load_state_dict(torch.load(f"{load_dir}/vae.pth"))
vae.to(device)
vae.eval()

mdn_rnn = MDNRNN(latent_dim=32, action_dim=3, hidden_dim=256, n_gaussians=5)
mdn_rnn.load_state_dict(torch.load(f"{load_dir}/mdn_rnn.pth"))
mdn_rnn.to(device)
mdn_rnn.eval()

controller = Controller()
controller.to(device)


# Load controller params
def set_controller_params(params):
    idx = 0
    for p in controller.parameters():
        size = p.numel()
        p.data = (
            torch.from_numpy(params[idx : idx + size])
            .float()
            .reshape(p.shape)
            .to(device)
        )
        idx += size


params = np.load(f"{load_dir}/controller_params.npy")
set_controller_params(params)
controller.eval()

print("Models loaded!")

# Storage
all_frames = []
all_actions = []
episode_rewards = []

env = gym.make("CarRacing-v3", continuous=True)

for episode in range(args.episodes):
    obs, _ = env.reset()
    obs = np.array(Image.fromarray(obs).resize((64, 64)))

    hidden = (torch.zeros(1, 1, 256).to(device), torch.zeros(1, 1, 256).to(device))
    episode_reward = 0

    for step in range(args.max_steps):
        all_frames.append(obs.copy())

        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float() / 255.0
            obs_t = obs_t.permute(2, 0, 1).unsqueeze(0).to(device)
            mu, _ = vae.encode(obs_t)
            z = mu.squeeze(0)
            h = hidden[0].squeeze(0).squeeze(0)

            action = controller(z, h).cpu().numpy()

        all_actions.append(action.copy())

        action_env = np.array([action[0], (action[1] + 1) / 2, (action[2] + 1) / 2])
        obs, reward, terminated, truncated, _ = env.step(action_env)
        obs = np.array(Image.fromarray(obs).resize((64, 64)))
        episode_reward += reward

        with torch.no_grad():
            z_input = z.unsqueeze(0).unsqueeze(0)
            a_input = (
                torch.from_numpy(action).float().unsqueeze(0).unsqueeze(0).to(device)
            )
            _, hidden = mdn_rnn(z_input, a_input, hidden)

        if terminated or truncated:
            break

    episode_rewards.append(episode_reward)
    print(f"Episode {episode+1}/{args.episodes}: {episode_reward:.1f}")

env.close()

all_frames = np.array(all_frames)
all_actions = np.array(all_actions)

print(f"\nCollected {len(all_frames)} frames")
print(f"Average reward: {np.mean(episode_rewards):.1f} ± {np.std(episode_rewards):.1f}")

np.save(f"{output_dir}/frames.npy", all_frames)
np.save(f"{output_dir}/actions.npy", all_actions)
print(f"Saved to {output_dir}/")
