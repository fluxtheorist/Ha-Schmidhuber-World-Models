import torch
import torch.nn as nn


# Controller
class ControllerVizDoom(nn.Module):
    # Initialization
    def __init__(self, latent_dim=64, hidden_dim=512):
        super().__init__()
        input_dim = latent_dim + hidden_dim + hidden_dim
        self.fc = nn.Linear(input_dim, 1, bias=False)

    # Forward pass
    def forward(self, z, h, c):
        x = torch.cat([z, h, c], dim=-1)
        out = torch.tanh(self.fc(x))
        val = out.squeeze()
        if val < -0.33:
            return 0
        elif val > 0.33:
            return 2
        else:
            return 1
