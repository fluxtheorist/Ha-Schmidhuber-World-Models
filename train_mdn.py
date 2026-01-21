import torch
import numpy as np
from vae import ConvVAE
from mdn_rnn import MDNRNN

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load trained VAE
vae = ConvVAE(latent_dim=32)
vae.load_state_dict(torch.load("outputs/vae.pth"))
vae.to(device)
vae.eval()

# Load data
frames = np.load("outputs/frames.npy")
actions = np.load("outputs/actions.npy")
print(f"Loadded {len(frames)} frames and {len(actions)} actions")

# Encode all frames through VAE to get z vectors
frames_tensor = torch.from_numpy(frames).float() / 255.0
frames_tensor = frames_tensor.permute(0, 3, 1, 2)

all_z = []
batch_size = 256

with torch.no_grad():
    for i in range(0, len(frames_tensor), batch_size):
        batch = frames_tensor[i : i + batch_size].to(device)
        mu, logvar = vae.encode(batch)
        all_z.append(mu.cpu())


all_z = torch.cat(all_z, dim=0)

# Convert actions to tensor
all_actions = torch.from_numpy(actions).float()

# Training params
SEQ_LEN = 50
EPISODE_LEN = 500
N_EPISODES = 50
EPOCHS = 20
BATCH_SIZE = 32

# Create MDN-RNN
model = MDNRNN(latent_dim=32, action_dim=3, hidden_dim=256, n_gaussians=5)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(EPOCHS):
    total_loss = 0
    n_batches = 0

    for _ in range(100):  # 100 batches per epoch
        batch_z = []
        batch_a = []
        batch_target = []

        for _ in range(BATCH_SIZE):
            # Pick random episode
            ep = np.random.randint(N_EPISODES)
            ep_start = ep * EPISODE_LEN

            # Pick random start within episode
            max_start = EPISODE_LEN - SEQ_LEN - 1
            start = np.random.randint(max_start)
            idx = ep_start + start

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

    print(f"Epoch {epoch + 1}, Loss: {total_loss / n_batches:.4f}")

torch.save(model.state_dict(), "outputs/mdn_rnn.pth")
