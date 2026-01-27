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

    def get_mdn_params(self, mdn_out):
        pi = mdn_out[:, :, :5]
        mu = mdn_out[:, :, 5:165]
        sigma = mdn_out[:, :, 165:]

        # Activations
        pi = torch.softmax(pi, dim=-1)
        sigma = torch.exp(sigma)

        # Reshape and match mu + sigma
        mu = mu.view(mu.size(0), mu.size(1), self.n_gaussians, self.latent_dim)
        sigma = sigma.view(
            sigma.size(0), sigma.size(1), self.n_gaussians, self.latent_dim
        )

        return pi, mu, sigma

    def sample(self, pi, mu, sigma):
        # Flatten for multinomial
        batch_size, seq_len, n_gaussians = pi.shape
        pi_flat = pi.view(-1, n_gaussians)

        # Sample one index per row
        indices = torch.multinomial(pi_flat, 1)
        indices = indices.view(batch_size, seq_len)

        # Add dimensions for gathering
        indices = indices.unsqueeze(-1).unsqueeze(-1)

        # Expand indices to match mu/sigma shape for gapthering
        indices = indices.expand(batch_size, seq_len, 1, self.latent_dim)

        # Gather the selected Gaussians's mu and sigma
        selected_mu = torch.gather(mu, 2, indices).squeeze(2)
        selected_sigma = torch.gather(sigma, 2, indices).squeeze(2)

        z = selected_mu + selected_sigma * torch.randn_like(selected_mu)

        return z

    def loss_function(self, pi, mu, sigma, target_z):
        # Expand target_z to match mu shape
        target_z = target_z.unsqueeze(2)

        # Gaussian log probability for each component
        log_prob = (
            -0.5 * torch.log(torch.tensor(2 * 3.13159))
            - torch.log(sigma)
            - 0.5 * ((target_z - mu) / sigma) ** 2
        )

        # Sum over latent dimensions
        log_prob = log_prob.sum(dim=-1)

        # Weight by pi and sum
        log_pi = torch.log(pi)
        log_weighted = log_prob + log_pi
        log_likelihood = torch.logsumexp(log_weighted, dim=-1)

        # Negative log likelihood averaged
        loss = -log_likelihood.mean()

        return loss


if __name__ == "__main__":
    model = MDNRNN(latent_dim=32, action_dim=3, hidden_dim=256)

    # Test with fake sequence
    batch_size = 4
    seq_len = 10
    z = torch.randn(batch_size, seq_len, 32)
    action = torch.randn(batch_size, seq_len, 3)

    mdn_out, hidden = model(z, action)
    pi, mu, sigma = model.get_mdn_params(mdn_out)
    z_sample = model.sample(pi, mu, sigma)
    target_z = torch.randn(batch_size, seq_len, 32)
    loss = model.loss_function(pi, mu, sigma, target_z)

    print(f"Input z: {z.shape}")
    print(f"Input action: {action.shape}")
    print(f"pi: {pi.shape}")
    print(f"mu: {mu.shape}")
    print(f"sigma: {sigma.shape}")
    print(f"Sampled z: {z_sample.shape}")
    print(f"Loss: {loss.item():.2f}")
