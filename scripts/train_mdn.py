import sys

sys.path.append("..")

import torch
import numpy as np
import argparse
from models.vae import ConvVAE
from models.mdn_rnn import MDNRNN

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


def encode_frames(frames, vae):
    frames_tensor = torch.from_numpy(frames).float() / 255.0
    frames_tensor = frames_tensor.permute(0, 3, 1, 2)

    all_z = []
    batch_size = 256

    with torch.no_grad():
        for i in range(0, len(frames_tensor), batch_size):
            batch = frames_tensor[i : i + batch_size].to(device)
            mu, _ = vae.encode(batch)
            all_z.append(mu.cpu())

    return torch.cat(all_z, dim=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=0, help="Iteration number")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    args = parser.parse_args()

    output_dir = f"../outputs/iter{args.iter}"

    # Load VAE from this iteration
    vae = ConvVAE(latent_dim=32)
    vae.load_state_dict(torch.load(f"{output_dir}/vae.pth"))
    vae.to(device)
    vae.eval()
    print(f"Loaded VAE from {output_dir}/vae.pth")

    # Load and encode all data up to this iteration
    all_z_list = []
    all_actions_list = []
    episode_lengths_list = []

    for i in range(args.iter + 1):
        iter_dir = f"../outputs/iter{i}"
        try:
            frames = np.load(f"{iter_dir}/frames.npy")
            actions = np.load(f"{iter_dir}/actions.npy")

            print(f"Encoding iter{i} ({len(frames)} frames)...")
            z = encode_frames(frames, vae)

            all_z_list.append(z)
            all_actions_list.append(torch.from_numpy(actions).float())

            # Try to load episode lengths, otherwise assume 500
            try:
                ep_lens = np.load(f"{iter_dir}/episode_lengths.npy")
                episode_lengths_list.append(ep_lens)
            except FileNotFoundError:
                # Assume 50 episodes of ~500 steps
                episode_lengths_list.append(np.array([500] * 50))

        except FileNotFoundError:
            print(f"No data found for iter{i}, skipping")

    all_z = torch.cat(all_z_list, dim=0)
    all_actions = torch.cat(all_actions_list, dim=0)
    all_episode_lengths = np.concatenate(episode_lengths_list)

    print(f"Total: {len(all_z)} encoded frames, {len(all_episode_lengths)} episodes")

    # Training params
    SEQ_LEN = 50
    BATCH_SIZE = 32

    # Build episode start indices
    episode_starts = [0]
    for length in all_episode_lengths[:-1]:
        episode_starts.append(episode_starts[-1] + length)
    episode_starts = np.array(episode_starts)
    n_episodes = len(episode_starts)

    # Create MDN-RNN
    model = MDNRNN(latent_dim=32, action_dim=3, hidden_dim=256, n_gaussians=5)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    for epoch in range(args.epochs):
        total_loss = 0
        n_batches = 0

        for _ in range(100):  # 100 batches per epoch
            batch_z = []
            batch_a = []
            batch_target = []

            for _ in range(BATCH_SIZE):
                # Pick random episode
                ep = np.random.randint(n_episodes)
                ep_start = episode_starts[ep]
                ep_len = all_episode_lengths[ep]

                # Pick random start within episode (need SEQ_LEN + 1 frames)
                if ep_len <= SEQ_LEN + 1:
                    continue  # Skip short episodes

                max_start = ep_len - SEQ_LEN - 1
                start = np.random.randint(max_start)
                idx = ep_start + start

                # Get sequence
                z_seq = all_z[idx : idx + SEQ_LEN]
                a_seq = all_actions[idx : idx + SEQ_LEN]
                target_seq = all_z[idx + 1 : idx + SEQ_LEN + 1]

                batch_z.append(z_seq)
                batch_a.append(a_seq)
                batch_target.append(target_seq)

            if len(batch_z) == 0:
                continue

            batch_z = torch.stack(batch_z).to(device)
            batch_a = torch.stack(batch_a).to(device)
            batch_target = torch.stack(batch_target).to(device)

            optimizer.zero_grad()
            mdn_out, _ = model(batch_z, batch_a)
            pi, mu, sigma = model.get_mdn_params(mdn_out)
            loss = model.loss_function(pi, mu, sigma, batch_target)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {total_loss / n_batches:.4f}")

    torch.save(model.state_dict(), f"{output_dir}/mdn_rnn.pth")
    print(f"Saved to {output_dir}/mdn_rnn.pth")
