#!/usr/bin/env python3
"""
Standalone MDN-RNN training for VizDoom, matching Ha's doomrnn.py + rnn_train.py.

This is a single-file translation of Ha's TensorFlow code to PyTorch.
No external model imports. Run directly.

Usage:
    python train_mdn_standalone.py --data_dir ../outputs/vizdoom/iter0

Key differences from the broken version:
1. Scalar action input (not one-hot) -> input_dim = 64 + 1 + 1 = 66
2. LSTMCell with manual loop (matches Ha's custom_rnn_autodecoder exactly)
3. Output layout: [restart_logit(1), gmm_params(960)] flattened then reshaped
4. GMM loss computed on FLAT tensors (batch*seq*latent, n_mix) like Ha
5. Learning rate decay matching Ha
"""

import sys
import os
import argparse
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("..")

# ============================================================
# MODEL (direct translation of Ha's doomrnn.py Model.build_model)
# ============================================================

RESTART_FACTOR = 10.0  # Ha's model_restart_factor


class MDNRNN(nn.Module):
    """MDN-RNN matching Ha's architecture exactly.

    Input per timestep: [z(64), action(1), restart(1)] = 66 dims
    LSTM hidden: 512
    Output: restart_logit(1) + GMM params(64*5*3 = 960) = 961
    """

    def __init__(self, z_size=64, n_mix=5, rnn_size=512):
        super().__init__()
        self.z_size = z_size  # WIDTH in Ha's code
        self.n_mix = n_mix  # KMIX
        self.rnn_size = rnn_size

        # Input: z + action_scalar + restart_flag = 66
        self.input_size = z_size + 1 + 1  # 66

        # LSTM cell (Ha uses tf.contrib.rnn.LayerNormBasicLSTMCell with layer_norm=False)
        self.lstm = nn.LSTMCell(self.input_size, rnn_size)

        # Output projection (Ha: output_w, output_b)
        # NOUT = WIDTH * KMIX * 3 + 1
        self.nout = z_size * n_mix * 3 + 1  # 961
        self.output_w = nn.Linear(rnn_size, self.nout)

    def forward_train(self, batch_z, batch_action, batch_restart, initial_state=None):
        """Training forward pass matching Ha's custom_rnn_autodecoder.

        Args:
            batch_z: (batch, seq_len+1, 64) - full sequence including target
            batch_action: (batch, seq_len+1) - scalar actions (0, 1, or 2 as float)
            batch_restart: (batch, seq_len+1) - restart flags
            initial_state: tuple of (h, c) each (batch, rnn_size), or None

        Returns:
            z_cost: scalar GMM loss
            r_cost: scalar restart loss
            final_state: (h, c) tuple
        """
        batch_size = batch_z.shape[0]
        LENGTH = batch_z.shape[1] - 1  # Ha: LENGTH = max_seq_len - 1
        device = batch_z.device

        # Input/target split (Ha lines 311-316)
        input_z = batch_z[:, :LENGTH, :]  # (batch, LENGTH, 64)
        input_action = batch_action[:, :LENGTH]  # (batch, LENGTH)
        input_restart = batch_restart[:, :LENGTH]  # (batch, LENGTH)
        target_z = batch_z[:, 1:, :]  # (batch, LENGTH, 64)
        target_restart = batch_restart[:, 1:]  # (batch, LENGTH)

        # Concatenate input: [z, action, restart] (Ha lines 318-320)
        input_seq = torch.cat(
            [
                input_z,
                input_action.unsqueeze(-1),  # (batch, LENGTH, 1)
                input_restart.unsqueeze(-1),  # (batch, LENGTH, 1)
            ],
            dim=2,
        )  # (batch, LENGTH, 66)

        # Initialize state
        if initial_state is None:
            h = torch.zeros(batch_size, self.rnn_size, device=device)
            c = torch.zeros(batch_size, self.rnn_size, device=device)
        else:
            h, c = initial_state

        zero_h = torch.zeros_like(h)
        zero_c = torch.zeros_like(c)

        # Custom RNN with restart (Ha lines 327-351)
        outputs = []
        for i in range(LENGTH):
            inp = input_seq[:, i, :]  # (batch, 66)

            # Reset state where restart > 0.5 (Ha lines 341-346)
            restart_flag = input_restart[:, i] > 0.5  # (batch,)
            if restart_flag.any():
                mask = restart_flag.unsqueeze(1)  # (batch, 1)
                c = torch.where(mask, zero_c, c)
                h = torch.where(mask, zero_h, h)

            h, c = self.lstm(inp, (h, c))
            outputs.append(h)

        # Stack outputs: (batch, LENGTH, rnn_size)
        rnn_output = torch.stack(outputs, dim=1)

        # Output projection (Ha lines 358-368)
        # Flatten to (batch*LENGTH, rnn_size), project, then reshape
        flat_output = rnn_output.reshape(-1, self.rnn_size)
        flat_output = self.output_w(flat_output)  # (batch*LENGTH, 961)

        # Split restart logits and GMM params (Ha lines 365-368)
        out_restart_logits = flat_output[:, 0]  # (batch*LENGTH,)
        gmm_output = flat_output[:, 1:]  # (batch*LENGTH, 960)

        # Reshape GMM to per-dimension: (batch*LENGTH*64, 15)
        gmm_output = gmm_output.reshape(-1, self.n_mix * 3)  # Ha line 368

        # Get MDN coefficients (Ha lines 381-384)
        logmix, mean, logstd = torch.chunk(gmm_output, 3, dim=1)  # each (B*L*64, 5)
        logmix = logmix - torch.logsumexp(logmix, dim=1, keepdim=True)

        # Flatten target (Ha line 393)
        flat_target = target_z.reshape(-1, 1)  # (batch*LENGTH*64, 1)

        # GMM loss (Ha lines 373-379)
        log_sqrt_2pi = np.log(np.sqrt(2.0 * np.pi))
        log_prob = (
            -0.5 * ((flat_target - mean) / torch.exp(logstd)) ** 2
            - logstd
            - log_sqrt_2pi
        )
        v = logmix + log_prob  # (B*L*64, 5)
        v = torch.logsumexp(v, dim=1, keepdim=True)  # (B*L*64, 1)
        z_cost = -torch.mean(v)

        # Restart loss (Ha lines 399-406)
        flat_target_restart = target_restart.reshape(-1, 1)  # (batch*LENGTH, 1)
        r_cost = F.binary_cross_entropy_with_logits(
            out_restart_logits.reshape(-1, 1), flat_target_restart, reduction="none"
        )
        factor = 1.0 + flat_target_restart * (RESTART_FACTOR - 1.0)
        r_cost = torch.mean(factor * r_cost)

        return z_cost, r_cost, (h.detach(), c.detach())

    def get_params_count(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# DATA LOADING (direct translation of Ha's rnn_train.py)
# ============================================================


def encode_all_chunks(vae_path, data_dir, device):
    """Encode all frame chunks with VAE, return mu and logvar per frame."""
    from models.vae import ConvVAE

    vae = ConvVAE(latent_dim=64).to(device)
    vae.load_state_dict(torch.load(vae_path, map_location=device))
    vae.eval()
    print("VAE loaded")

    all_mu = []
    all_logvar = []
    all_actions = []
    all_dones = []
    chunk_idx = 1

    while True:
        frames_path = f"{data_dir}/frames_chunk{chunk_idx}.npy"
        if not os.path.exists(frames_path):
            break

        print(f"  Encoding chunk {chunk_idx}...")
        frames = np.load(frames_path)
        actions = np.load(f"{data_dir}/actions_chunk{chunk_idx}.npy")
        dones = np.load(f"{data_dir}/dones_chunk{chunk_idx}.npy")

        frames_tensor = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0

        chunk_mu = []
        chunk_logvar = []
        with torch.no_grad():
            for i in range(0, len(frames_tensor), 256):
                batch = frames_tensor[i : i + 256].to(device)
                mu, logvar = vae.encode(batch)
                chunk_mu.append(mu.cpu().numpy())
                chunk_logvar.append(logvar.cpu().numpy())

        all_mu.append(np.concatenate(chunk_mu, axis=0))
        all_logvar.append(np.concatenate(chunk_logvar, axis=0))
        all_actions.append(actions.astype(np.float32))
        all_dones.append(dones)

        del frames, frames_tensor
        chunk_idx += 1

    del vae

    all_mu = np.concatenate(all_mu, axis=0)
    all_logvar = np.concatenate(all_logvar, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)
    all_dones = np.concatenate(all_dones, axis=0)

    return all_mu, all_logvar, all_actions, all_dones


def split_into_episodes(mu, logvar, actions, dones):
    """Split data into per-episode lists. Returns list of [mu, logvar, action]."""
    episodes = []
    ep_start = 0
    for i in range(len(dones)):
        if dones[i] == 1:
            episodes.append(
                [
                    mu[ep_start : i + 1],
                    logvar[ep_start : i + 1],
                    actions[ep_start : i + 1],
                ]
            )
            ep_start = i + 1
    if ep_start < len(dones):
        episodes.append([mu[ep_start:], logvar[ep_start:], actions[ep_start:]])
    return episodes


def create_batches(all_data, batch_size, seq_length, N_z=64):
    """Exact translation of Ha's create_batches (rnn_train.py lines 76-106).

    Concatenates shuffled episodes, marks restart, reshapes into batches.
    """
    random.shuffle(all_data)

    num_frames = sum(len(d[0]) for d in all_data)
    num_batches = num_frames // (batch_size * seq_length)
    num_frames_adjusted = num_batches * batch_size * seq_length

    if num_batches == 0:
        raise ValueError(
            f"Not enough data: {num_frames} frames for batch={batch_size} seq={seq_length}"
        )

    # Ha uses float16 for mu/logvar and uint8 for restart
    data_mu = np.zeros((num_frames, N_z), dtype=np.float16)
    data_logvar = np.zeros((num_frames, N_z), dtype=np.float16)
    data_action = np.zeros(num_frames, dtype=np.float16)
    data_restart = np.zeros(num_frames, dtype=np.uint8)

    idx = 0
    for mu, logvar, action in all_data:
        N = len(action)
        data_mu[idx : idx + N] = mu.reshape(N, N_z)
        data_logvar[idx : idx + N] = logvar.reshape(N, N_z)
        data_action[idx : idx + N] = action.reshape(N)
        data_restart[idx] = 1  # Mark first frame as restart
        idx += N

    # Truncate
    data_mu = data_mu[:num_frames_adjusted]
    data_logvar = data_logvar[:num_frames_adjusted]
    data_action = data_action[:num_frames_adjusted]
    data_restart = data_restart[:num_frames_adjusted]

    # Reshape and split (Ha lines 101-104)
    data_mu = np.split(data_mu.reshape(batch_size, -1, N_z), num_batches, axis=1)
    data_logvar = np.split(
        data_logvar.reshape(batch_size, -1, N_z), num_batches, axis=1
    )
    data_action = np.split(data_action.reshape(batch_size, -1), num_batches, axis=1)
    data_restart = np.split(data_restart.reshape(batch_size, -1), num_batches, axis=1)

    return data_mu, data_logvar, data_action, data_restart


def get_batch(batch_idx, data_mu, data_logvar, data_action, data_restart):
    """Exact translation of Ha's get_batch (rnn_train.py lines 108-115).

    Key: samples z from mu + exp(logvar/2) * randn (stochastic!)
    """
    batch_mu = data_mu[batch_idx]
    batch_logvar = data_logvar[batch_idx]
    batch_action = data_action[batch_idx]
    batch_restart = data_restart[batch_idx]
    batch_s = batch_logvar.shape
    batch_z = batch_mu + np.exp(batch_logvar / 2.0) * np.random.randn(*batch_s)
    return (
        batch_z.astype(np.float32),
        batch_action.astype(np.float32),
        batch_restart.astype(np.float32),
    )


# ============================================================
# TRAINING (direct translation of Ha's rnn_train.py training loop)
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../outputs/vizdoom/iter0")
    parser.add_argument("--output", type=str, default="../outputs/vizdoom/iter0")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument(
        "--seq_length",
        type=int,
        default=500,
        help="max_seq_len in Ha's code. Actual LSTM steps = seq_length-1",
    )
    args = parser.parse_args()

    # Ha's hyperparameters (hardcoded to match his defaults)
    rnn_size = 512
    n_mix = 5
    z_size = 64
    grad_clip = 1.0
    learning_rate = 0.001
    decay_rate = 0.99999
    min_learning_rate = 0.00001
    restart_factor = 10.0

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load and encode data
    vae_path = f"{args.data_dir}/vae.pth"
    all_mu, all_logvar, all_actions, all_dones = encode_all_chunks(
        vae_path, args.data_dir, device
    )
    print(f"Total frames: {len(all_mu)}")

    all_data = split_into_episodes(all_mu, all_logvar, all_actions, all_dones)
    print(f"Total episodes: {len(all_data)}")
    ep_lens = [len(d[0]) for d in all_data]
    print(
        f"Episode lengths: mean={np.mean(ep_lens):.0f}, min={min(ep_lens)}, max={max(ep_lens)}"
    )

    # Save initial z (Ha rnn_train.py lines 124-132)
    initial_mu = []
    initial_logvar = []
    for i in range(min(1000, len(all_data))):
        initial_mu.append(all_data[i][0][0])
        initial_logvar.append(all_data[i][1][0])
    np.savez(
        f"{args.output}/initial_z.npz",
        mu=np.array(initial_mu),
        logvar=np.array(initial_logvar),
    )
    print(f"Saved {len(initial_mu)} initial z vectors")

    # Free the raw encoded data (episodes list is enough)
    del all_mu, all_logvar, all_actions, all_dones

    # Create model
    model = MDNRNN(z_size=z_size, n_mix=n_mix, rnn_size=rnn_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Model params: {model.get_params_count():,}")

    # ===== SANITY CHECK: train on one batch for 100 steps =====
    print("\n=== Sanity check: can z_cost drop on a single batch? ===")
    sanity_data = create_batches(all_data, args.batch_size, args.seq_length, z_size)
    sanity_z, sanity_a, sanity_r = get_batch(0, *sanity_data)
    sanity_z_t = torch.from_numpy(sanity_z).to(device)
    sanity_a_t = torch.from_numpy(sanity_a).to(device)
    sanity_r_t = torch.from_numpy(sanity_r).to(device)

    sanity_opt = torch.optim.Adam(model.parameters(), lr=0.001)
    for i in range(100):
        z_cost, r_cost, _ = model.forward_train(sanity_z_t, sanity_a_t, sanity_r_t)
        loss = z_cost + r_cost
        sanity_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
        sanity_opt.step()
        if i % 10 == 0:
            print(
                f"  sanity step {i:3d}: z_cost={z_cost.item():.6f}, r_cost={r_cost.item():.6f}"
            )

    if z_cost.item() > 1.41:
        print("  WARNING: z_cost did NOT drop after 100 steps on same batch!")
        print("  This suggests a fundamental issue with the loss or device.")
        print("  Trying on CPU...")
        model_cpu = MDNRNN(z_size=z_size, n_mix=n_mix, rnn_size=rnn_size)
        opt_cpu = torch.optim.Adam(model_cpu.parameters(), lr=0.001)
        sz = sanity_z_t.cpu()
        sa = sanity_a_t.cpu()
        sr = sanity_r_t.cpu()
        for i in range(100):
            z_cost_cpu, r_cost_cpu, _ = model_cpu.forward_train(sz, sa, sr)
            loss_cpu = z_cost_cpu + r_cost_cpu
            opt_cpu.zero_grad()
            loss_cpu.backward()
            torch.nn.utils.clip_grad_value_(model_cpu.parameters(), 1.0)
            opt_cpu.step()
            if i % 10 == 0:
                print(f"  CPU step {i:3d}: z_cost={z_cost_cpu.item():.6f}")
        if z_cost_cpu.item() < 1.3:
            print("  CPU works! MPS is the problem. Switching to CPU.")
            device = torch.device("cpu")
            model = model_cpu
            optimizer = opt_cpu
        else:
            print("  CPU also stuck. There's a bug in the loss computation.")
            sys.exit(1)
    else:
        print(f"  Sanity check PASSED. z_cost dropped to {z_cost.item():.4f}")

    # Re-initialize model fresh for actual training
    model = MDNRNN(z_size=z_size, n_mix=n_mix, rnn_size=rnn_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print(f"\nFresh model on {device}, starting training...")

    # Training loop (Ha rnn_train.py lines 140-170)
    global_step = 0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        print(f"\nPreparing data for epoch {epoch}")
        data_mu, data_logvar, data_action, data_restart = create_batches(
            all_data, args.batch_size, args.seq_length, z_size
        )
        num_batches = len(data_mu)

        end = time.time()
        print(f"  {num_batches} batches, data prep took {end - start:.1f}s")
        start = time.time()

        # Initialize hidden state for epoch (Ha line 150)
        batch_state = None  # Will be initialized in model

        epoch_z_cost = 0
        epoch_r_cost = 0

        for local_step in range(num_batches):
            # Get batch (Ha line 154)
            batch_z, batch_action, batch_restart = get_batch(
                local_step, data_mu, data_logvar, data_action, data_restart
            )

            # Learning rate decay (Ha line 156)
            curr_lr = (learning_rate - min_learning_rate) * (
                decay_rate**global_step
            ) + min_learning_rate
            for pg in optimizer.param_groups:
                pg["lr"] = curr_lr

            # Convert to tensors
            t_z = torch.from_numpy(batch_z).to(device)
            t_action = torch.from_numpy(batch_action).to(device)
            t_restart = torch.from_numpy(batch_restart).to(device)

            # Forward (includes input/target split, loss computation)
            z_cost, r_cost, batch_state = model.forward_train(
                t_z, t_action, t_restart, batch_state
            )
            cost = z_cost + r_cost

            # Backward with gradient clipping (Ha lines 414-416)
            optimizer.zero_grad()
            cost.backward()

            # Log gradient norms periodically for debugging
            if global_step % 100 == 0 and global_step > 0:
                fc_grad = model.output_w.weight.grad
                gmm_grad_norm = fc_grad[1:].norm().item()
                restart_grad_norm = fc_grad[0].norm().item()
                lstm_grad_norm = sum(
                    p.grad.norm().item()
                    for p in model.lstm.parameters()
                    if p.grad is not None
                )
                print(
                    f"  [grad] gmm_w: {gmm_grad_norm:.6f}, restart_w: {restart_grad_norm:.6f}, lstm: {lstm_grad_norm:.6f}"
                )

            torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            optimizer.step()

            epoch_z_cost += z_cost.item()
            epoch_r_cost += r_cost.item()
            global_step += 1

            # Log every 20 steps (Ha line 165)
            if global_step % 20 == 0 and global_step > 0:
                end = time.time()
                time_taken = end - start
                start = time.time()
                print(
                    f"  step: {global_step}, lr: {curr_lr:.6f}, "
                    f"cost: {cost.item():.4f}, z_cost: {z_cost.item():.4f}, "
                    f"r_cost: {r_cost.item():.4f}, time: {time_taken:.2f}s"
                )

        avg_z = epoch_z_cost / num_batches
        avg_r = epoch_r_cost / num_batches
        print(
            f"Epoch {epoch}/{args.epochs} | z_cost: {avg_z:.4f} | r_cost: {avg_r:.4f} | "
            f"total: {avg_z + avg_r:.4f} | steps: {global_step}"
        )

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"{args.output}/mdn_rnn.pth")
            print(f"  Checkpoint saved")

    # Final save
    torch.save(model.state_dict(), f"{args.output}/mdn_rnn.pth")
    print(f"\nTraining complete. Model saved to {args.output}/mdn_rnn.pth")
    print(f"Final z_cost: {avg_z:.4f}, r_cost: {avg_r:.4f}")
