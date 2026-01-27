"""
Collect data using trained controller.
Step 1 of iterative training.
"""

import torch
import numpy as np
import gymnasium as gym
from PIL import Image
from vae import ConvVAE
from mdn_rnn import MDNRNN
from controller import Controller

# Settings
NUM_EPISODES = 50
MAX_STEPS = 500
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load models
vae = ConvVAE(latent_dim=32)
vae.load_state_dict(torch.load("outputs/vae.pth"))
vae.to(device)
vae.eval()

mdn_rnn = MDNRNN(latent_dim=32, action_dim=3, hidden_dim=256, n_gaussians=5)
mdn_rnn.load_state_dict(torch.load("outputs/mdn_rnn.pth"))
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


params = np.load("outputs/controller_params.npy")
set_controller_params(params)
controller.eval()

print("Models loaded!")

# Storage
all_frames = []
all_actions = []
episode_rewards = []

env = gym.make("CarRacing-v3", continuous=True)

for episode in range(NUM_EPISODES):
    obs, _ = env.reset()
    obs = np.array(Image.fromarray(obs).resize((64, 64)))

    hidden = (torch.zeros(1, 1, 256).to(device), torch.zeros(1, 1, 256).to(device))
    episode_reward = 0

    for step in range(MAX_STEPS):
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
    print(f"Episode {episode+1}/{NUM_EPISODES}: {episode_reward:.1f}")

env.close()

all_frames = np.array(all_frames)
all_actions = np.array(all_actions)

print(f"\nCollected {len(all_frames)} frames")
print(f"Average reward: {np.mean(episode_rewards):.1f} ± {np.std(episode_rewards):.1f}")

np.save("outputs/frames_iter1.npy", all_frames)
np.save("outputs/actions_iter1.npy", all_actions)
print("Saved to outputs/frames_iter1.npy and actions_iter1.npy")
