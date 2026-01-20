import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from vae import ConvVAE

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
frames = np.load("outputs/frames.npy")
print(f"Loaded {len(frames)} frames")

# Convert to torch tensor and noramlize
frames_tensor = torch.from_numpy(frames).float() / 255.0

# Reorder [N, H, W, C] -> [N, C, H, W]
frames_tensor = frames_tensor.permute(0, 3, 1, 2)

print(f"Tensor shape: {frames_tensor.shape}")

# DataLoader
dataloader = DataLoader(frames_tensor, batch_size=32, shuffle=True)

# Create model and optimizer
model = ConvVAE(latent_dim=32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
EPOCHS = 50

for epoch in range(EPOCHS):
    total_loss = 0

    for batch in dataloader:
        batch = batch.to(device)
        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        recon, mu, logvar = model(batch)

        # Compute loss
        loss, recon_loss, kl_loss = model.loss_function(recon, batch, mu, logvar)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        # Track total loss
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.2f}")

# Save model
torch.save(model.state_dict(), "outputs/vae.pth")
