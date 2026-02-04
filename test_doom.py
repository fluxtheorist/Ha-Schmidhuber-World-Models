import torch
import numpy as np
import matplotlib.pyplot as plt
from models.vae import ConvVAE

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load model
vae = ConvVAE(latent_dim=64)
vae.load_state_dict(torch.load("outputs/vizdoom/iter0/vae.pth"))
vae.to(device)
vae.eval()

# Load a chunk and grab some random frames
frames = np.load("outputs/vizdoom/iter0/frames_chunk1.npy")
indices = np.random.choice(len(frames), 5, replace=False)
samples = frames[indices]

# Encode and decode
samples_tensor = torch.from_numpy(samples).float() / 255.0
samples_tensor = samples_tensor.permute(0, 3, 1, 2).to(device)

with torch.no_grad():
    recon, mu, logvar = vae(samples_tensor)
    recon = recon.permute(0, 2, 3, 1).cpu().numpy()

# Plot
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i in range(5):
    axes[0, i].imshow(samples[i])
    axes[0, i].set_title("Original")
    axes[0, i].axis("off")

    axes[1, i].imshow(recon[i])
    axes[1, i].set_title("Reconstruction")
    axes[1, i].axis("off")

plt.tight_layout()
plt.savefig("outputs/vizdoom/iter0/vae_reconstruction.png")
plt.show()
print("Saved to outputs/vizdoom/iter0/vae_reconstruction.png")
