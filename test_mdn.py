# test_mdn.py
import torch
import numpy as np
from vae import ConvVAE
from mdn_rnn import MDNRNN

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load VAE
vae = ConvVAE(latent_dim=32)
vae.load_state_dict(torch.load("outputs/vae.pth"))
vae.to(device)
vae.eval()

# Load MDN-RNN
model = MDNRNN(latent_dim=32, action_dim=3, hidden_dim=256, n_gaussians=5)
model.load_state_dict(torch.load("outputs/mdn_rnn.pth"))
model.to(device)
model.eval()

# Load data and encode
frames = np.load("outputs/frames.npy")
actions = np.load("outputs/actions.npy")

frames_tensor = torch.from_numpy(frames).float() / 255.0
frames_tensor = frames_tensor.permute(0, 3, 1, 2)

with torch.no_grad():
    all_z = []
    for i in range(0, len(frames_tensor), 256):
        batch = frames_tensor[i : i + 256].to(device)
        mu, _ = vae.encode(batch)
        all_z.append(mu.cpu())
    all_z = torch.cat(all_z, dim=0)

all_actions = torch.from_numpy(actions).float()

# Test predictions
with torch.no_grad():
    test_z = all_z[0:50].unsqueeze(0).to(device)
    test_a = all_actions[0:50].unsqueeze(0).to(device)
    actual_next_z = all_z[1:51]

    mdn_out, _ = model(test_z, test_a)
    pi, mu, sigma = model.get_mdn_params(mdn_out)

    # Sampled prediction
    sampled_z = model.sample(pi, mu, sigma)
    sampled_mse = ((sampled_z.squeeze(0).cpu() - actual_next_z) ** 2).mean()

    # Mean prediction
    pi_expanded = pi.unsqueeze(-1)
    mean_pred = (pi_expanded * mu).sum(dim=2)
    mean_mse = ((mean_pred.squeeze(0).cpu() - actual_next_z) ** 2).mean()

    # Baseline
    baseline_mse = ((test_z.squeeze(0).cpu() - actual_next_z) ** 2).mean()

    print(f"Sampled prediction MSE: {sampled_mse:.4f}")
    print(f"Mean prediction MSE: {mean_mse:.4f}")
    print(f"Baseline MSE (no change): {baseline_mse:.4f}")
