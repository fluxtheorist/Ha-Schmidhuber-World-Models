"""
Train MDN-RNN for VizDoom, matching Ha's rnn_train.py approach.

Key differences from previous version:
1. Predicts "restart" (first frame of episode) instead of "death" (last frame)
2. Restart flag fed as input to LSTM; hidden state reset at restart
3. Z sampled from mu+logvar each batch (stochastic)
4. Episodes concatenated, split into long sequences (seq_len=500)
5. Hidden state carries across batches within epoch
6. restart_factor=10 weighting
7. Gradient clipping (1.0)
"""

import sys

sys.path.append("..")

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
import random
import time

from models.vae import ConvVAE
from models.mdn_rnn import MDNRNN

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


def encode_all_chunks(vae, data_dir, device):
    """Encode all frame chunks with VAE, return mu and logvar per episode."""
    all_mu = []
    all_logvar = []
    all_actions = []
    all_dones = []
    chunk_idx = 1

    while True:
        frames_path = f"{data_dir}/frames_chunk{chunk_idx}.npy"
        if not os.path.exists(frames_path):
            break

        print(f"Encoding chunk {chunk_idx}...")
        frames = np.load(frames_path)
        actions = np.load(f"{data_dir}/actions_chunk{chunk_idx}.npy")
        dones = np.load(f"{data_dir}/dones_chunk{chunk_idx}.npy")

        frames_tensor = torch.from_numpy(frames).float() / 255.0
        frames_tensor = frames_tensor.permute(0, 3, 1, 2).to(device)

        chunk_mu = []
        chunk_logvar = []
        with torch.no_grad():
            for i in range(0, len(frames_tensor), 256):
                batch = frames_tensor[i : i + 256]
                mu, logvar = vae.encode(batch)
                chunk_mu.append(mu.cpu())
                chunk_logvar.append(logvar.cpu())

        all_mu.append(torch.cat(chunk_mu, dim=0))
        all_logvar.append(torch.cat(chunk_logvar, dim=0))
        all_actions.append(torch.from_numpy(actions))
        all_dones.append(torch.from_numpy(dones))

        del frames, frames_tensor
        chunk_idx += 1

    all_mu = torch.cat(all_mu, dim=0)
    all_logvar = torch.cat(all_logvar, dim=0)
    all_actions = torch.cat(all_actions, dim=0)
    all_dones = torch.cat(all_dones, dim=0)

    return all_mu, all_logvar, all_actions, all_dones


def split_into_episodes(mu, logvar, actions, dones):
    """Split concatenated data into per-episode lists.

    Returns list of (mu, logvar, action) tuples, one per episode.
    """
    episodes = []
    ep_start = 0

    for i in range(len(dones)):
        if dones[i] == 1:
            ep_mu = mu[ep_start : i + 1]
            ep_logvar = logvar[ep_start : i + 1]
            ep_action = actions[ep_start : i + 1]
            episodes.append((ep_mu, ep_logvar, ep_action))
            ep_start = i + 1

    # Handle trailing data without a done
    if ep_start < len(dones):
        episodes.append((mu[ep_start:], logvar[ep_start:], actions[ep_start:]))

    return episodes


def create_batches(episodes, batch_size, seq_length, latent_dim):
    """Create batches matching Ha's create_batches function.

    Concatenates shuffled episodes, marks restart at episode starts,
    then reshapes into (batch_size, num_batches, seq_length) chunks.
    """
    random.shuffle(episodes)

    # Count total frames
    total_frames = sum(len(ep[0]) for ep in episodes)
    num_batches = total_frames // (batch_size * seq_length)
    num_frames_adjusted = num_batches * batch_size * seq_length

    if num_batches == 0:
        raise ValueError(
            f"Not enough data for batch_size={batch_size}, seq_length={seq_length}"
        )

    # Allocate arrays
    data_mu = np.zeros((total_frames, latent_dim), dtype=np.float32)
    data_logvar = np.zeros((total_frames, latent_dim), dtype=np.float32)
    data_action = np.zeros(total_frames, dtype=np.float32)
    data_restart = np.zeros(total_frames, dtype=np.float32)

    idx = 0
    for ep_mu, ep_logvar, ep_action in episodes:
        N = len(ep_mu)
        data_mu[idx : idx + N] = ep_mu.numpy()
        data_logvar[idx : idx + N] = ep_logvar.numpy()
        data_action[idx : idx + N] = ep_action.numpy().astype(np.float32)
        data_restart[idx] = 1.0  # Mark first frame of episode as restart
        idx += N

    # Truncate to fit evenly
    data_mu = data_mu[:num_frames_adjusted]
    data_logvar = data_logvar[:num_frames_adjusted]
    data_action = data_action[:num_frames_adjusted]
    data_restart = data_restart[:num_frames_adjusted]

    # Reshape: (batch_size, total_steps_per_batch) then split into chunks
    data_mu = np.split(data_mu.reshape(batch_size, -1, latent_dim), num_batches, axis=1)
    data_logvar = np.split(
        data_logvar.reshape(batch_size, -1, latent_dim), num_batches, axis=1
    )
    data_action = np.split(data_action.reshape(batch_size, -1), num_batches, axis=1)
    data_restart = np.split(data_restart.reshape(batch_size, -1), num_batches, axis=1)

    return data_mu, data_logvar, data_action, data_restart, num_batches


def get_batch(batch_idx, data_mu, data_logvar, data_action, data_restart):
    """Get a batch, sampling z from mu+logvar (matches Ha's get_batch)."""
    mu = data_mu[batch_idx]
    logvar = data_logvar[batch_idx]
    action = data_action[batch_idx]
    restart = data_restart[batch_idx]

    # Sample z from VAE posterior (stochastic!) - this is key
    z = mu + np.exp(logvar / 2.0) * np.random.randn(*mu.shape)

    return z.astype(np.float32), action, restart


def save_initial_z(episodes, save_path, n=1000):
    """Save initial z mu/logvar for dream environment initialization."""
    initial_mu = []
    initial_logvar = []
    for i in range(min(n, len(episodes))):
        mu = episodes[i][0][0].numpy()  # First frame mu
        logvar = episodes[i][1][0].numpy()  # First frame logvar
        initial_mu.append(mu)
        initial_logvar.append(logvar)

    np.savez(save_path, mu=np.array(initial_mu), logvar=np.array(initial_logvar))
    print(f"Saved {len(initial_mu)} initial z vectors to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../outputs/vizdoom/iter0")
    parser.add_argument("--output", type=str, default="../outputs/vizdoom/iter0")
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Training epochs (Ha uses 400, we use fewer)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=100, help="Batch size (Ha uses 100)"
    )
    parser.add_argument(
        "--seq_length", type=int, default=500, help="Sequence length (Ha uses 500)"
    )
    parser.add_argument(
        "--restart_factor", type=float, default=10.0, help="Weight for restart loss"
    )
    parser.add_argument(
        "--grad_clip", type=float, default=1.0, help="Gradient clipping"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load VAE
    vae = ConvVAE(latent_dim=64).to(device)
    vae.load_state_dict(torch.load(f"{args.data_dir}/vae.pth"))
    vae.eval()
    print("VAE loaded")

    # Encode all data
    all_mu, all_logvar, all_actions, all_dones = encode_all_chunks(
        vae, args.data_dir, device
    )
    print(f"Total frames: {len(all_mu)}")

    # Split into episodes
    episodes = split_into_episodes(all_mu, all_logvar, all_actions, all_dones)
    print(f"Total episodes: {len(episodes)}")
    ep_lengths = [len(ep[0]) for ep in episodes]
    print(
        f"Episode lengths: mean={np.mean(ep_lengths):.1f}, min={np.min(ep_lengths)}, max={np.max(ep_lengths)}"
    )

    # Save initial z for dream environment
    save_initial_z(episodes, f"{args.output}/initial_z.npz")

    # Create model
    model = MDNRNN(latent_dim=64, action_dim=3, hidden_dim=512, n_gaussians=5).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"MDNRNN params: {total_params:,}")

    # Training loop
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        # Recreate batches each epoch (reshuffles episodes)
        data_mu, data_logvar, data_action, data_restart, num_batches = create_batches(
            episodes, args.batch_size, args.seq_length, 64
        )

        # Initialize hidden state for the epoch (carries across batches!)
        hidden = model.init_hidden(args.batch_size, device)

        total_z_cost = 0
        total_r_cost = 0

        for batch_idx in range(num_batches):
            # Get batch with stochastic z sampling
            batch_z, batch_action, batch_restart = get_batch(
                batch_idx, data_mu, data_logvar, data_action, data_restart
            )

            # Convert to tensors
            # Input: z[:, :-1], action[:, :-1], restart[:, :-1]
            # Target: z[:, 1:], restart[:, 1:]
            z_tensor = torch.from_numpy(batch_z).to(device)
            action_tensor = torch.from_numpy(batch_action).long().to(device)
            restart_tensor = torch.from_numpy(batch_restart).to(device)

            input_z = z_tensor[:, :-1, :]
            input_action = action_tensor[:, :-1]
            input_restart = restart_tensor[:, :-1]

            target_z = z_tensor[:, 1:, :]
            target_restart = restart_tensor[:, 1:]

            # Detach hidden state (don't backprop through previous batches)
            hidden = (hidden[0].detach(), hidden[1].detach())

            # Forward
            output, hidden = model(input_z, input_action, input_restart, hidden)
            restart_logits, logmix, mean, logstd = model.get_mdn_params(output)

            # Losses
            z_cost = model.loss_function(logmix, mean, logstd, target_z)
            r_cost = model.restart_loss(
                restart_logits, target_restart, args.restart_factor
            )
            loss = z_cost + r_cost

            # Backward with gradient clipping
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), args.grad_clip)
            optimizer.step()

            total_z_cost += z_cost.item()
            total_r_cost += r_cost.item()

        elapsed = time.time() - start_time
        avg_z = total_z_cost / num_batches
        avg_r = total_r_cost / num_batches
        print(
            f"Epoch {epoch}/{args.epochs} | z_cost: {avg_z:.4f} | r_cost: {avg_r:.4f} | total: {avg_z + avg_r:.4f} | batches: {num_batches} | time: {elapsed:.1f}s"
        )

    # Save model
    torch.save(model.state_dict(), f"{args.output}/mdn_rnn.pth")
    print(f"Model saved to {args.output}/mdn_rnn.pth")
