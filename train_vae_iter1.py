"""
Train VAE on combined data from iteration 0 and 1
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from vae import ConvVAE

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load BOTH datasets
frames_iter0 = np.load("outputs/frames.npy")
frames_iter1 = np.load("outputs/frames_iter1.npy")

print(f"Iteration 0 frames: {len(frames_iter0)}")
print(f"Iteration 1 frames: {len(frames_iter1)}")

# Combine them
all_frames = np.concatenate([frames_iter0, frames_iter1], axis=0)
print(f"Total frames: {len(all_frames)}")

# Prepare data
frames_tensor = torch.from_numpy(all_frames).float() / 255.0
frames_tensor = frames_tensor.permute(0, 3, 1, 2)

dataloader = DataLoader(frames_tensor, batch_size=32, shuffle=True)

# Train VAE
model = ConvVAE(latent_dim=32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 10

for epoch in range(EPOCHS):
    total_loss = 0
    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(batch)
        loss, _, _ = model.loss_function(recon, batch, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.0f}")

torch.save(model.state_dict(), "outputs/vae_iter1.pth")
print("Saved to outputs/vae_iter1.pth")
