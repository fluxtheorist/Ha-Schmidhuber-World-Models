import torch
import torch.nn as nn


class ConvVAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()

        # Encoder layers
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1
        )
        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1
        )
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

        # Decoder layer
        self.fc_decode = nn.Linear(latent_dim, 256 * 4 * 4)
        self.transpose1 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1
        )
        self.transpose2 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1
        )
        self.transpose3 = nn.ConvTranspose2d(
            in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1
        )
        self.transpose4 = nn.ConvTranspose2d(
            in_channels=32, out_channels=3, kernel_size=4, stride=2, padding=1
        )
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        # Pass first layer
        h = self.relu(self.conv1(x))

        # Pass second layer
        h = self.relu(self.conv2(h))

        # Pass third slayer
        h = self.relu(self.conv3(h))

        # Pass fourth layer
        h = self.relu(self.conv4(h))

        # Flatten the conv output
        h = h.view(h.size(0), -1)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(h.size(0), 256, 4, 4)

        h = self.relu(self.transpose1(h))
        h = self.relu(self.transpose2(h))
        h = self.relu(self.transpose3(h))
        h = self.sigmoid(self.transpose4(h))

        return h

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


if __name__ == "__main__":
    # This only runs when you execute vae.py directly
    model = ConvVAE(latent_dim=32)
    fake_image = torch.randn(1, 3, 64, 64)
    recon, mu, logvar = model(fake_image)
    print(f"Input: {fake_image.shape}")
    print(f"Reconstruction: {recon.shape}")
    print(f"mu: {mu.shape}, logvar: {logvar.shape}")
