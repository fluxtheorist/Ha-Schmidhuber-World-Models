import torch
import torch.nn as nn
import numpy as np
from vae import ConvVAE
from reward_predictor import RewardPredictor

# Device selection
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load VAE
vae = ConvVAE(latent_dim=32)
vae.load_state_dict(torch.load("outputs/vae.pth"))
vae.to(device)
vae.eval()

# Load data
frames = np.load("outputs/frames.npy")
rewards = np.load("outputs/rewards.npy")
print(f"Loaded {len(frames)} frames and {len(rewards)} rewards")
print(f"Reward range: {rewards.min():.2f} to {rewards.max():.2f}")

# Encode all frames to z
frames_tensor = torch.from_numpy(frames).float() / 255.0
frames_tensor = frames_tensor.permute(0, 3, 1, 2)

all_z = []
with torch.no_grad():
    for i in range(0, len(frames_tensor), 256):
        batch = frames_tensor[i : i + 256].to(device)
        mu, _ = vae.encode(batch)
        all_z.append(mu.cpu())

all_z = torch.cat(all_z, dim=0)
all_rewards = torch.from_numpy(rewards).float()

print(f"Encoded z: {all_z.shape}")

# Train reward predictor
model = RewardPredictor(latent_dim=32, hidden_dim=64)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 50
BATCH_SIZE = 256

for epoch in range(EPOCHS):
    total_loss = 0
    n_batches = 0

    # Shuffle
    perm = torch.randperm(len(all_z))

    for i in range(0, len(all_z), BATCH_SIZE):
        idx = perm[i : i + BATCH_SIZE]
        z_batch = all_z[idx].to(device)
        r_batch = all_rewards[idx].to(device)

        optimizer.zero_grad()
        pred = model(z_batch)
        loss = nn.functional.mse_loss(pred, r_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {total_loss/n_batches:.4f}")

torch.save(model.state_dict(), "outputs/reward_predictor.pth")
print("Saved reward predictor")

# Quick test
model.eval()
with torch.no_grad():
    test_z = all_z[:10].to(device)
    test_r = all_rewards[:10]
    pred_r = model(test_z).cpu()

    print("\nActual vs Predicted:")
    for actual, pred in zip(test_r, pred_r):
        print(f"  {actual:.2f} vs {pred:.2f}")
