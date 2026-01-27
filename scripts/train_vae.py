import sys

sys.path.append("..")

import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse
from models.vae import ConvVAE

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=0, help="Iteration number")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    args = parser.parse_args()

    output_dir = f"../outputs/iter{args.iter}"

    # Load all data up to this iteration
    all_frames = []
    for i in range(args.iter + 1):
        iter_dir = f"../outputs/iter{i}"
        try:
            frames = np.load(f"{iter_dir}/frames.npy")
            all_frames.append(frames)
            print(f"Loaded {len(frames)} frames from iter{i}")
        except FileNotFoundError:
            print(f"No frames found at {iter_dir}, skipping")

    if not all_frames:
        print("No data found! Run collect_data.py first.")
        exit(1)

    all_frames = np.concatenate(all_frames, axis=0)
    print(f"Total frames: {len(all_frames)}")

    # Convert to torch tensor and normalize
    frames_tensor = torch.from_numpy(all_frames).float() / 255.0

    # Reorder [N, H, W, C] -> [N, C, H, W]
    frames_tensor = frames_tensor.permute(0, 3, 1, 2)

    print(f"Tensor shape: {frames_tensor.shape}")

    # DataLoader
    dataloader = DataLoader(frames_tensor, batch_size=32, shuffle=True)

    # Create model and optimizer
    model = ConvVAE(latent_dim=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    for epoch in range(args.epochs):
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

        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {total_loss:.2f}")

    # Save model
    torch.save(model.state_dict(), f"{output_dir}/vae.pth")
    print(f"Saved to {output_dir}/vae.pth")
