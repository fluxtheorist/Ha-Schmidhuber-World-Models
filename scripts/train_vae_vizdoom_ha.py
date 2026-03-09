#!/usr/bin/env python3
"""
Train VAE for VizDoom - streaming version that loads one chunk at a time.

Usage:
    python train_vae_vizdoom.py --data_dir ../outputs/vizdoom/iter1
    python train_vae_vizdoom.py --data_dirs ../outputs/vizdoom/iter0 ../outputs/vizdoom/iter1
"""

import sys

sys.path.append("..")

import os
import argparse
import time
import glob
import numpy as np
import torch

from models.vae import ConvVAE

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dirs",
        type=str,
        nargs="+",
        default=["../outputs/vizdoom/iter0"],
        help="One or more data directories",
    )
    parser.add_argument(
        "--data_dir", type=str, default=None, help="Single data dir (shorthand)"
    )
    parser.add_argument("--output", type=str, default="../outputs/vizdoom/iter1")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--kl_tolerance", type=float, default=0.5)
    parser.add_argument("--latent_dim", type=int, default=64)
    args = parser.parse_args()

    if args.data_dir:
        args.data_dirs = [args.data_dir]

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Find all chunk files
    chunk_paths = []
    for d in args.data_dirs:
        paths = sorted(glob.glob(os.path.join(d, "frames_chunk*.npy")))
        chunk_paths.extend(paths)
        print(f"  {d}: {len(paths)} chunks")
    print(f"Total chunks: {len(chunk_paths)}")

    # Count total frames without loading
    total_frames = 0
    chunk_sizes = []
    for p in chunk_paths:
        with open(p, "rb") as f:
            version = np.lib.format.read_magic(f)
            if version[0] == 1:
                shape, _, _ = np.lib.format.read_array_header_1_0(f)
            else:
                shape, _, _ = np.lib.format.read_array_header_2_0(f)
        chunk_sizes.append(shape[0])
        total_frames += shape[0]
    print(f"Total frames: {total_frames:,}")

    num_batches_per_epoch = total_frames // args.batch_size
    print(f"Batches per epoch: {num_batches_per_epoch}")

    # Create model
    vae = ConvVAE(latent_dim=args.latent_dim, kl_tolerance=args.kl_tolerance).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr)
    params = sum(p.numel() for p in vae.parameters())
    print(f"VAE params: {params:,}")

    # Training loop - stream one chunk at a time
    global_step = 0
    print("step | loss | recon_loss | kl_loss")

    for epoch in range(1, args.epochs + 1):
        chunk_order = np.random.permutation(len(chunk_paths))
        epoch_loss = 0
        epoch_recon = 0
        epoch_kl = 0
        epoch_batches = 0
        start = time.time()

        for ci in chunk_order:
            frames = np.load(chunk_paths[ci])
            np.random.shuffle(frames)

            n_batches = len(frames) // args.batch_size
            for idx in range(n_batches):
                batch = frames[idx * args.batch_size : (idx + 1) * args.batch_size]
                obs = (
                    torch.from_numpy(batch).float().permute(0, 3, 1, 2).to(device)
                    / 255.0
                )

                recon, mu, logvar = vae(obs)
                loss, recon_loss, kl_loss = vae.loss_function(recon, obs, mu, logvar)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                global_step += 1
                epoch_batches += 1
                epoch_loss += loss.item()
                epoch_recon += recon_loss.item()
                epoch_kl += kl_loss.item()

                if global_step % 500 == 0:
                    print(
                        f"  step {global_step}: loss={loss.item():.2f}, "
                        f"recon={recon_loss.item():.2f}, kl={kl_loss.item():.2f}"
                    )

            del frames

        elapsed = time.time() - start
        avg_loss = epoch_loss / epoch_batches
        avg_recon = epoch_recon / epoch_batches
        avg_kl = epoch_kl / epoch_batches
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
    sample = np.load(chunk_paths[0])[:1000]
    sample_t = torch.from_numpy(sample).float().permute(0, 3, 1, 2).to(device) / 255.0
    with torch.no_grad():
        mu, logvar = vae.encode(sample_t)
    print(f"mu: mean={mu.mean():.4f}, std={mu.std():.4f}")
    print(f"logvar: mean={logvar.mean():.4f}, std={logvar.std():.4f}")
    print(f"mu std across batch (should be >> 0.01): {mu.std(dim=0).mean():.4f}")
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(dim=0)
    print(f"KL per dim (should be ~0.5): mean={kl_per_dim.mean():.4f}")
