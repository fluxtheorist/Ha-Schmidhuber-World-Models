#!/usr/bin/env python3
"""
Train VAE for VizDoom, matching Ha's vae_train.py.

Key: kl_tolerance=0.5 prevents posterior collapse.

Usage:
    python train_vae_vizdoom.py --data_dir ../outputs/vizdoom/iter0
"""

import sys

sys.path.append("..")

import os
import argparse
import time
import numpy as np
import torch

from models.vae import ConvVAE

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../outputs/vizdoom/iter0")
    parser.add_argument("--output", type=str, default="../outputs/vizdoom/iter0")
    parser.add_argument("--epochs", type=int, default=10, help="Ha uses 10")
    parser.add_argument("--batch_size", type=int, default=100, help="Ha uses 100")
    parser.add_argument("--lr", type=float, default=0.0001, help="Ha uses 0.0001")
    parser.add_argument("--kl_tolerance", type=float, default=0.5, help="Ha uses 0.5")
    parser.add_argument("--latent_dim", type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load all frames
    print("Loading frames...")
    all_frames = []
    chunk_idx = 1
    while True:
        path = f"{args.data_dir}/frames_chunk{chunk_idx}.npy"
        if not os.path.exists(path):
            break
        print(f"  Loading chunk {chunk_idx}...")
        frames = np.load(path)
        all_frames.append(frames)
        chunk_idx += 1

    dataset = np.concatenate(all_frames, axis=0)
    del all_frames
    print(f"Total frames: {len(dataset)}")

    num_batches = len(dataset) // args.batch_size
    print(f"Batches per epoch: {num_batches}")

    # Create model (with kl_tolerance!)
    vae = ConvVAE(latent_dim=args.latent_dim, kl_tolerance=args.kl_tolerance).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr)

    params = sum(p.numel() for p in vae.parameters())
    print(f"VAE params: {params:,}")

    # Training loop (Ha vae_train.py lines 96-112)
    global_step = 0
    print("step | loss | recon_loss | kl_loss")

    for epoch in range(1, args.epochs + 1):
        np.random.shuffle(dataset)
        epoch_loss = 0
        epoch_recon = 0
        epoch_kl = 0
        start = time.time()

        for idx in range(num_batches):
            batch = dataset[idx * args.batch_size : (idx + 1) * args.batch_size]
            obs = torch.from_numpy(batch).float().permute(0, 3, 1, 2).to(device) / 255.0

            recon, mu, logvar = vae(obs)
            loss, recon_loss, kl_loss = vae.loss_function(recon, obs, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()

            if global_step % 500 == 0:
                print(
                    f"  step {global_step}: loss={loss.item():.2f}, "
                    f"recon={recon_loss.item():.2f}, kl={kl_loss.item():.2f}"
                )

        elapsed = time.time() - start
        avg_loss = epoch_loss / num_batches
        avg_recon = epoch_recon / num_batches
        avg_kl = epoch_kl / num_batches
        print(
            f"Epoch {epoch}/{args.epochs} | loss: {avg_loss:.2f} | "
            f"recon: {avg_recon:.2f} | kl: {avg_kl:.2f} | time: {elapsed:.1f}s"
        )

    # Save
    torch.save(vae.state_dict(), f"{args.output}/vae.pth")
    print(f"\nVAE saved to {args.output}/vae.pth")

    # Verify latent quality
    print("\nVerifying latent quality...")
    vae.eval()
    sample_frames = dataset[:1000]
    sample_t = (
        torch.from_numpy(sample_frames).float().permute(0, 3, 1, 2).to(device) / 255.0
    )
    with torch.no_grad():
        mu, logvar = vae.encode(sample_t)
    print(f"mu: mean={mu.mean():.4f}, std={mu.std():.4f}")
    print(f"logvar: mean={logvar.mean():.4f}, std={logvar.std():.4f}")
    print(f"mu std across batch (should be >> 0.01): {mu.std(dim=0).mean():.4f}")
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(dim=0)
    print(
        f"KL per dim (should be ~0.5): mean={kl_per_dim.mean():.4f}, "
        f"min={kl_per_dim.min():.4f}, max={kl_per_dim.max():.4f}"
    )
