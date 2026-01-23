import torch
import time
import numpy as np
from vae import ConvVAE
from mdn_rnn import MDNRNN
from controller import Controller
from dream_env import DreamEnv

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
controller.load_state_dict(torch.load("outputs/controller.pth"))
controller.to(device)
controller.eval()

# Load initial z vectors from real data
frames = np.load("outputs/frames.npy")
frames_tensor = torch.from_numpy(frames).float() / 255.0
frames_tensor = frames_tensor.permute(0, 3, 1, 2)

with torch.no_grad():
    all_z = []
    for i in range(0, len(frames_tensor), 256):
        batch = frames_tensor[i : i + 256].to(device)
        mu, _ = vae.encode(batch)
        all_z.append(mu.cpu())
    all_z = torch.cat(all_z, dim=0)

print(f"Loaded {len(all_z)} initial z vectors")

# Create dream environment
dream = DreamEnv(mdn_rnn, all_z, device, temperature=1.0)

# Run 10 dream episodes and time it
n_episodes = 10
start = time.time()

for ep in range(n_episodes):
    z, h = dream.reset()
    total_reward = 0

    for step in range(500):
        action = controller(z, h)
        z, h, reward, done = dream.step(action)
        total_reward += reward

        if done:
            break

    print(f"Dream episode {ep+1}: reward={total_reward:.1f}, steps={dream.steps}")

elapsed = time.time() - start
print(f"\n{n_episodes} dream episodes in {elapsed:.2f} seconds")
print(f"That's {elapsed/n_episodes:.3f} seconds per episode")
