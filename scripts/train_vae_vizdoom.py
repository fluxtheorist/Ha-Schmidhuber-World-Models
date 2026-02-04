import sys

sys.path.append("..")

import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
from models.vae import ConvVAE

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../outputs/vizdoom/iter0",
        help="Data directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../outputs/vizdoom/iter0",
        help="Output directory",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--latent_dim", type=int, default=64, help="Latent dimensions")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Model and Optimizer
    model = ConvVAE(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(f"VAE with latent_dim={args.latent_dim}")

    # Chunk-based Training
    for epoch in range(args.epochs):
        total_loss = 0
        total_recon = 0
        total_kl = 0
        total_batches = 0
        chunk_idx = 1

        while True:
            chunk_path = f"{args.data_dir}/frames_chunk{chunk_idx}.npy"
            if not os.path.exists(chunk_path):
                break

            frames = np.load(chunk_path)
            frames_tensor = torch.from_numpy(frames).float() / 255.0
            frames_tensor = frames_tensor.permute(0, 3, 1, 2)
            dataloader = DataLoader(
                frames_tensor, batch_size=args.batch_size, shuffle=True
            )

            for batch in dataloader:
                batch = batch.to(device)
                optimizer.zero_grad()

                recon, mu, logvar = model(batch)
                loss, recon_loss, kl_loss = model.loss_function(recon, batch, mu, logvar)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_kl += kl_loss.item()
                total_batches += 1

            del frames, frames_tensor, dataloader
            chunk_idx += 1

        avg_loss = total_loss / total_batches
        avg_recon = total_recon / total_batches
        avg_kl = total_kl / total_batches
        print(f"Epoch {epoch + 1}/{args.epochs} | Loss={avg_loss:.2f} | Recon={avg_recon:.2f} | KL={avg_kl:.2f}")

    torch.save(model.state_dict(), f"{args.output}/vae.pth")
    print(f"Model saved to {args.output}/vae.pth")
