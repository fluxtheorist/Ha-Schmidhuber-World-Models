import torch
import numpy as np
import matplotlib.pyplot as plt
from vae import ConvVAE

# Load model
model = ConvVAE(latent_dim=32)
model.load_state_dict(torch.load("outputs/vae.pth"))
model.eval()

# Load some frames
frames = np.load("outputs/frames.npy")
frames_tensor = torch.from_numpy(frames[:5]).float() / 255.0
frames_tensor = frames_tensor.permute(0, 3, 1, 2)

# Reconstructions
with torch.no_grad():
    recon, mu, logvar = model(frames_tensor)

# Plot original vs reconstruction
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for i in range(5):
    # Original
    axes[0, i].imshow(frames_tensor[i].permute(1, 2, 0).numpy())
    axes[0, i].set_title("Original")
    axes[0, i].axis("off")

    # Reconstruction
    axes[1, i].imshow(recon[i].permute(1, 2, 0).numpy())
    axes[1, i].set_title("Reconstruction")
    axes[1, i].axis("off")

plt.tight_layout()
plt.savefig("outputs/reconstruction_comparison.png")
plt.show()
