import sys

sys.path.append("..")

import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
from models.vae import ConvVAE
from models.mdn_rnn import MDNRNN

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=64,
        help="Latent dimension for loss normalization",
    )
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
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--sequence_length", type=int, default=50, help="Sequence length"
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load VAE
    vae = ConvVAE(latent_dim=64).to(device)
    vae.load_state_dict(torch.load(f"{args.data_dir}/vae.pth"))
    vae.eval()
    print("VAE loaded")

    # Collect latent vectors
    all_z = []
    all_actions = []
    all_dones = []
    chunk_idx = 1

    while True:
        frames_path = f"{args.data_dir}/frames_chunk{chunk_idx}.npy"
        if not os.path.exists(frames_path):
            break

        print(f"Encoding chunk {chunk_idx}")

        # Load chunk data
        frames = np.load(frames_path)
        actions = np.load(f"{args.data_dir}/actions_chunk{chunk_idx}.npy")
        dones = np.load(f"{args.data_dir}/dones_chunk{chunk_idx}.npy")

        # Encode frames to z
        frames_tensor = torch.from_numpy(frames).float() / 255.0
        frames_tensor = frames_tensor.permute(0, 3, 1, 2).to(device)
        with torch.no_grad():
            chunk_z = []
            for i in range(0, len(frames_tensor), 256):
                batch = frames_tensor[i : i + 256]
                mu, _ = vae.encode(batch)
                chunk_z.append(mu.cpu())
            chunk_z = torch.cat(chunk_z, dim=0)

        # Append to all arrays
        all_z.append(chunk_z)
        all_actions.append(torch.from_numpy(actions).long())
        all_dones.append(torch.from_numpy(dones).float())

        # Cleanup
        del frames, frames_tensor
        chunk_idx += 1

    # Concatenate all data
    all_z = torch.cat(all_z, dim=0)
    all_actions = torch.cat(all_actions, dim=0)
    all_dones = torch.cat(all_dones, dim=0)
    print(f"Total encoded frames: {len(all_z)}")

    # Model
    model = MDNRNN(latent_dim=64, action_dim=3, hidden_dim=512, n_gaussians=5).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(f"MDNRNN: latent_dim=64, action_dim=3, hidden_dim=512, n_gaussians=5")

    # Find death frame indices for oversampling
    death_indices = torch.where(all_dones == 1)[0].numpy()
    print(
        f"Death frames: {len(death_indices)} ({100*len(death_indices)/len(all_dones):.2f}%)"
    )

    # Training loop
    max_start_idx = len(all_z) - args.sequence_length - 1
    # Precompute valid starting indices (sequences that don't cross episode boundaries)
    print("Finding valid sequence starts...")
    valid_starts = []
    death_set = set(death_indices.tolist())
    for idx in range(max_start_idx):
        # Check if any frame in positions [idx, idx+seq_len-2] is a death
        # (death at the last position idx+seq_len-1 is OK)
        chunk = all_dones[idx : idx + args.sequence_length - 1]
        if chunk.sum() == 0:
            valid_starts.append(idx)
    valid_starts = np.array(valid_starts)
    print(
        f"Valid starts: {len(valid_starts)} / {max_start_idx} ({100*len(valid_starts)/max_start_idx:.1f}%)"
    )

    # Also find valid starts that end in death (for oversampling)
    death_starts = []
    for d_idx in death_indices:
        idx = max(0, d_idx - args.sequence_length + 1)
        idx = min(idx, max_start_idx)
        chunk = all_dones[idx : idx + args.sequence_length - 1]
        if chunk.sum() == 0:
            death_starts.append(idx)
    death_starts = np.array(death_starts)
    print(f"Death-anchored valid starts: {len(death_starts)}")
    for epoch in range(args.epochs):
        total_mdn_loss = 0
        death_loss = 0
        num_batches = 0

        for _ in range(100):
            z_batch = []
            a_batch = []
            d_batch = []
            target_z_batch = []

            for _ in range(args.batch_size):
                if np.random.random() < 0.5 and len(death_starts) > 0:
                    idx = death_starts[np.random.randint(len(death_starts))]
                else:
                    idx = valid_starts[np.random.randint(len(valid_starts))]

                z_batch.append(all_z[idx : idx + args.sequence_length])
                a_batch.append(all_actions[idx : idx + args.sequence_length])
                d_batch.append(all_dones[idx + 1 : idx + args.sequence_length + 1])
                target_z_batch.append(all_z[idx + 1 : idx + args.sequence_length + 1])

            z_batch = torch.stack(z_batch).to(device)
            a_batch = torch.stack(a_batch).to(device)
            d_batch = torch.stack(d_batch).to(device)
            target_z_batch = torch.stack(target_z_batch).to(device)

            optimizer.zero_grad()
            mdn_out, death_logits, _ = model(z_batch, a_batch)
            pi, mu, sigma = model.get_mdn_params(mdn_out)

            mdn_loss = model.loss_function(pi, mu, sigma, target_z_batch)
            d_loss = model.death_loss(death_logits, d_batch)
            # Weight death loss to compensate for class imbalance
            # (~0.5% of frames are deaths, so ~200x weight balances it)
            loss = (mdn_loss + d_loss) / args.latent_dim  # normalize by LSIZE

            loss.backward()
            optimizer.step()

            total_mdn_loss += mdn_loss.item()
            death_loss += d_loss.item()
            num_batches += 1

        print(
            f"Epoch {epoch + 1}/{args.epochs} | Mdn: {total_mdn_loss / num_batches:.2f} | Death: {death_loss / num_batches:.4f}"
        )

    torch.save(model.state_dict(), f"{args.output}/mdn_rnn.pth")
    print(f"Model saved to {args.output}/mdn_rnn.pth")
