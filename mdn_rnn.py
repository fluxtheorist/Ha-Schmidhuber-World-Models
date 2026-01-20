import torch
import torch.nn as nn


class MDNRNN(nn.Module):
    def __init__(self, latent_dim=32, action_dim=3, hidden_dim=256, n_gaussians=5):
        super().__init__()

        # Takes concatenated action and z as inputs
        input_dim = latent_dim + action_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_gaussians = n_gaussians

        # MDN ouput
        mdn_output_dim = n_gaussians * (1 + 2 * latent_dim)
        self.fc = nn.Linear(hidden_dim, mdn_output_dim)

    def forward(self, z, action, hidden=None):
        x = torch.cat([z, action], dim=-1)

        # Pass through LSTM
        lstm_out, hidden = self.lstm(x, hidden)

        # Pass through MDN linear layer
        mdn_out = self.fc(lstm_out)

        return mdn_out, hidden


if __name__ == "__main__":
    model = MDNRNN(latent_dim=32, action_dim=3, hidden_dim=256)

    # Test with fake sequence
    batch_size = 4
    seq_len = 10
    z = torch.randn(batch_size, seq_len, 32)
    action = torch.randn(batch_size, seq_len, 3)

    output, hidden = model(z, action)
    print(f"Input z: {z.shape}")
    print(f"Input action: {action.shape}")
    print(f"LSTM output: {output.shape}")
    print(f"Hidden state: {hidden[0].shape}")
