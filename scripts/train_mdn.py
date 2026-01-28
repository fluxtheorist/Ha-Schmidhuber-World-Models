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

    total_frames = 0
    for i in range(args.iter + 1):
        iter_dir = f"../outputs/iter{i}"
        try:
            frames = np.load(f"{iter_dir}/frames.npy")
            actions = np.load(f"{iter_dir}/actions.npy")

            print(f"Encoding iter{i} ({len(frames)} frames)...")
            z = encode_frames(frames, vae)

            all_z_list.append(z)
            all_actions_list.append(torch.from_numpy(actions).float())
            total_frames += len(frames)

        except FileNotFoundError:
            print(f"No data found for iter{i}, skipping")

    all_z = torch.cat(all_z_list, dim=0)
    all_actions = torch.cat(all_actions_list, dim=0)

    print(f"Total: {len(all_z)} encoded frames")

    # Training params
    SEQ_LEN = 50
    BATCH_SIZE = 32

    # Create MDN-RNN
    model = MDNRNN(latent_dim=32, action_dim=3, hidden_dim=256, n_gaussians=5)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Simple approach: just sample random contiguous sequences from ALL data
    # This ignores episode boundaries but works fine in practice
    max_start_idx = len(all_z) - SEQ_LEN - 1
    print(f"Sampling sequences from indices 0 to {max_start_idx}")

    # Training loop
    for epoch in range(args.epochs):
        total_loss = 0
        n_batches = 0

        for _ in range(100):  # 100 batches per epoch
            batch_z = []
            batch_a = []
            batch_target = []

            for _ in range(BATCH_SIZE):
                # Pick random start index
                idx = np.random.randint(0, max_start_idx)

                # Get sequence
                z_seq = all_z[idx : idx + SEQ_LEN]
                a_seq = all_actions[idx : idx + SEQ_LEN]
                target_seq = all_z[idx + 1 : idx + SEQ_LEN + 1]

                batch_z.append(z_seq)
                batch_a.append(a_seq)
                batch_target.append(target_seq)

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
