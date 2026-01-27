import torch
import torch.nn as nn


class RewardPredictor(nn.Module):
    def __init__(self, latent_dim=32, hidden_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)


if __name__ == "__main__":
    model = RewardPredictor()
    z = torch.randn(10, 32)
    pred = model(z)
    print(f"Input: {z.shape}")
    print(f"Output: {pred.shape}")
