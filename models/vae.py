import torch
import torch.nn as nn


class ConvVAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()

        # Encoder: kernel=4, stride=2, NO padding (paper architecture)
        self.conv1 = nn.Conv2d(3, 32, 4, stride=2)  # 64 -> 31
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)  # 31 -> 14
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)  # 14 -> 6
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)  # 6 -> 2
        # Flatten: 2*2*256 = 1024
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 1024)
        self.deconv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)  # 1 -> 5
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)  # 5 -> 13
        self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)  # 13 -> 30
        self.deconv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)  # 30 -> 64

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        h = self.relu(self.conv3(h))
        h = self.relu(self.conv4(h))
        h = h.reshape(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(h.size(0), 1024, 1, 1)
        h = self.relu(self.deconv1(h))
        h = self.relu(self.deconv2(h))
        h = self.relu(self.deconv3(h))
        h = self.sigmoid(self.deconv4(h))
        return h

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def loss_function(self, recon, x, mu, logvar):
        recon_loss = nn.functional.mse_loss(recon, x, reduction="sum")
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_loss, recon_loss, kl_loss


if __name__ == "__main__":
    # This only runs when you execute vae.py directly
    model = ConvVAE(latent_dim=32)
    fake_image = torch.randn(1, 3, 64, 64)
    recon, mu, logvar = model(fake_image)

    total_loss, recon_loss, kl_loss = model.loss_function(recon, fake_image, mu, logvar)

    print(f"Input: {fake_image.shape}")
    print(f"Reconstruction: {recon.shape}")
    print(f"Total loss: {total_loss.item():.2f}")
    print(f"Recon loss: {recon_loss.item():.2f}, KL loss: {kl_loss.item():.2f}")
