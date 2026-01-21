import torch
import numpy as np
from vae import ConvVAE
from mdn_rnn import MDNRNN

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load trained VAE
vae = ConvVAE(latent_dim=32)
vae.load_state_dict(torch.load("outputs/vae.pth"))
vae.to(device)
vae.eval()

# Load data
frames = np.load("outputs/frames.npy")
actions = np.load("outputs/actions.npy")
print(f"Loadded {len(frames)} frames and {len(actions)} actions")

# Encode all frames through VAE to get z vectors
frames_tensor = torch.from_numpy(frames).float() / 255.0
frames_tensor = frames_tensor.permute(0, 3, 1, 2)

all_z = []
batch_size = 256

with torch.grad():
    for i in range(0, len(frames_tensor), batch_size):
        batch = frames_tensor[i : i + batch_size].to(device)
        mu, logvar = vae.encode(batch)
        all_z.append(mu.cpu())


all_z = torch.car(all_z, dim=0)
print(f"Encoded z shape: {all_z.shape}")
