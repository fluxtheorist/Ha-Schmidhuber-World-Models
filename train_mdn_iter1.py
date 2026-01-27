"""
Train MDN-RNN on combined data from iteration 0 and 1
"""

import torch
import numpy as np
from vae import ConvVAE
from mdn_rnn import MDNRNN

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load the NEW VAE
vae = ConvVAE(latent_dim=32)
vae.load_state_dict(torch.load("outputs/vae_iter1.pth"))
vae.to(device)
vae.eval()

# Load both datasets
frames_iter0 = np.load("outputs/frames.npy")
actions_iter0 = np.load("outputs/actions.npy")
frames_iter1 = np.load("outputs/frames_iter1.npy")
actions_iter1 = np.load("outputs/actions_iter1.npy")

print(f"Iter 0: {len(frames_iter0)} frames")
print(f"Iter 1: {len(frames_iter1)} frames")


# Encode all frames through VAE
def encode_frames(frames):
    frames_t = torch.from_numpy(frames).float() / 255.0
    frames_t = frames_t.permute(0, 3, 1, 2)

    all_z = []
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(frames_t), batch_size):
            batch = frames_t[i : i + batch_size].to(device)
            mu, _ = vae.encode(batch)
            all_z.append(mu.cpu())

    return torch.cat(all_z, dim=0).numpy()


print("Encoding iteration 0...")
z_iter0 = encode_frames(frames_iter0)
print("Encoding iteration 1...")
z_iter1 = encode_frames(frames_iter1)

# Combine
all_z = np.concatenate([z_iter0, z_iter1], axis=0)
all_actions = np.concatenate([actions_iter0, actions_iter1], axis=0)
print(f"Total: {len(all_z)} encoded frames")

# Create sequences
seq_len = 50
n_seqs = len(all_z) // seq_len

z_seqs = all_z[: n_seqs * seq_len].reshape(n_seqs, seq_len, 32)
a_seqs = all_actions[: n_seqs * seq_len].reshape(n_seqs, seq_len, 3)

print(f"Created {n_seqs} sequences of length {seq_len}")

z_tensor = torch.from_numpy(z_seqs).float()
a_tensor = torch.from_numpy(a_seqs).float()

# Train MDN-RNN
model = MDNRNN(latent_dim=32, action_dim=3, hidden_dim=256, n_gaussians=5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 20
batch_size = 32

for epoch in range(EPOCHS):
    total_loss = 0
    indices = np.random.permutation(n_seqs)

    for i in range(0, n_seqs, batch_size):
        idx = indices[i : i + batch_size]
        z_batch = z_tensor[idx].to(device)
        a_batch = a_tensor[idx].to(device)

        optimizer.zero_grad()

        mdn_out, _ = model(z_batch[:, :-1], a_batch[:, :-1])
        pi, mu, sigma = model.get_mdn_params(mdn_out)

        loss = model.loss_function(pi, mu, sigma, z_batch[:, 1:])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.2f}")

torch.save(model.state_dict(), "outputs/mdn_rnn_iter1.pth")
print("Saved to outputs/mdn_rnn_iter1.pth")
