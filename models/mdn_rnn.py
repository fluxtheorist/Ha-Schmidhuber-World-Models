import torch
import torch.nn as nn
import torch.nn.functional as F


class MDNRNN(nn.Module):
    def __init__(self, latent_dim=32, action_dim=3, hidden_dim=256, n_gaussians=5):
        super().__init__()

        input_dim = latent_dim + action_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_gaussians = n_gaussians

        # MDN output
        mdn_output_dim = n_gaussians * (1 + 2 * latent_dim)
        self.fc_mdn = nn.Linear(hidden_dim, mdn_output_dim)

        # Death prediction head
        self.fc_death = nn.Linear(hidden_dim, 1)

    def forward(self, z, action, hidden=None):
        # One-hot encode discrete actions if needed
        if action.dim() == 2:
            action = F.one_hot(action.long(), num_classes=3).float()

        x = torch.cat([z, action], dim=-1)

        lstm_out, hidden = self.lstm(x, hidden)

        mdn_out = self.fc_mdn(lstm_out)
        death_logits = self.fc_death(lstm_out)

        return mdn_out, death_logits, hidden

    def get_mdn_params(self, mdn_out):
        n_g = self.n_gaussians
        lat = self.latent_dim

        pi = mdn_out[:, :, :n_g]
        mu = mdn_out[:, :, n_g : n_g + n_g * lat]
        sigma = mdn_out[:, :, n_g + n_g * lat :]

        pi = torch.softmax(pi, dim=-1)
        sigma = torch.exp(sigma)

        mu = mu.view(mu.size(0), mu.size(1), n_g, lat)
        sigma = sigma.view(sigma.size(0), sigma.size(1), n_g, lat)

        return pi, mu, sigma

    def death_loss(self, death_logits, target_death):
        death_logits = death_logits.squeeze(-1)
        return F.binary_cross_entropy_with_logits(death_logits, target_death)

    def sample(self, pi, mu, sigma):
        batch_size, seq_len, n_gaussians = pi.shape
        pi_flat = pi.view(-1, n_gaussians)

        indices = torch.multinomial(pi_flat, 1)
        indices = indices.view(batch_size, seq_len)
        indices = indices.unsqueeze(-1).unsqueeze(-1)
        indices = indices.expand(batch_size, seq_len, 1, self.latent_dim)

        selected_mu = torch.gather(mu, 2, indices).squeeze(2)
        selected_sigma = torch.gather(sigma, 2, indices).squeeze(2)

        z = selected_mu + selected_sigma * torch.randn_like(selected_mu)
        return z

    def loss_function(self, pi, mu, sigma, target_z):
        target_z = target_z.unsqueeze(2)

        log_prob = (
            -0.5 * torch.log(torch.tensor(2 * 3.14159))
            - torch.log(sigma)
            - 0.5 * ((target_z - mu) / sigma) ** 2
        )

        log_prob = log_prob.sum(dim=-1)
        log_pi = torch.log(pi + 1e-8)
        log_weighted = log_prob + log_pi
        log_likelihood = torch.logsumexp(log_weighted, dim=-1)

        return -log_likelihood.mean()
