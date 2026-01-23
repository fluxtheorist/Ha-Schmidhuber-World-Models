import torch
import numpy as np
import gymnasium as gym
from PIL import Image
from vae import ConvVAE
from mdn_rnn import MDNRNN
from controller import Controller

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
# Random weights - not trained yet

# Run one episode
env = gym.make("CarRacing-v3", continuous=True, render_mode="human")
obs, _ = env.reset()
obs = np.array(Image.fromarray(obs).resize((64, 64)))

hidden = (torch.zeros(1, 1, 256).to(device), torch.zeros(1, 1, 256).to(device))

total_reward = 0

for step in range(500):
    with torch.no_grad():
        # V: obs -> z
        obs_tensor = torch.from_numpy(obs).float() / 255.0
        obs_tensor = obs_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
        mu, _ = vae.encode(obs_tensor)
        z = mu.squeeze(0)

        # M: get h
        h = hidden[0].squeeze(0).squeeze(0)

        # C: [z, h] -> action
        action = controller(z, h).cpu().numpy()

        action_env = np.array([action[0], (action[1] + 1) / 2, (action[2] + 1) / 2])

    obs, reward, terminated, truncated, _ = env.step(action_env)
    obs = np.array(Image.fromarray(obs).resize((64, 64)))
    total_reward += reward

    # Update hidden state
    with torch.no_grad():
        z_input = z.unsqueeze(0).unsqueeze(0)
        a_input = torch.from_numpy(action).float().unsqueeze(0).unsqueeze(0).to(device)
        _, hidden = mdn_rnn(z_input, a_input, hidden)

    if terminated or truncated:
        break

env.close()
print(f"Total reward: {total_reward:.1f}")
