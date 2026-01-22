import torch
import torch.nn as nn


class Controller(nn.Module):
    def __init__(self, latent_dim=32, hidden_dim=256, action_dim=3):
        super().__init__()

        input_dim = latent_dim + hidden_dim
        self.fc = nn.Linear(input_dim, action_dim)

    def forward(self, z, h):
        x = torch.cat([z, h], dim=-1)
        action = self.fc(x)

        # Squash to validate action ranges
        action = torch.tanh(action)

        return action


if __name__ == "__main__":
    controller = Controller()

    # Count parameters
    n_params = sum(p.numel() for p in controller.parameters())
    print(f"Controller parameters: {n_params}")

    # Tests
    z = torch.randn(32)
    h = torch.randn(256)
    action = controller(z, h)
    print(f"Action shape: {action.shape}")
    print(f"Action: {action}")
