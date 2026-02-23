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

        # Combined output: GMM params + reward + terminal
        # (2 * latent_dim + 1) * n_gaussians for GMM + 1 reward + 1 terminal
        output_dim = (2 * latent_dim + 1) * n_gaussians + 2
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z, action, hidden=None):
        # One-hot encode discrete actions if needed
        if action.dim() == 2:
            action = F.one_hot(action.long(), num_classes=3).float()

        x = torch.cat([z, action], dim=-1)

        lstm_out, hidden = self.lstm(x, hidden)

        out = self.fc_out(lstm_out)

        # Split: GMM params, reward, terminal
        gmm_out = out[:, :, :-2]
        reward_out = out[:, :, -2]  # not used for VizDoom but matches reference
        death_logits = out[:, :, -1:]

        return gmm_out, death_logits, hidden

    def get_mdn_params(self, mdn_out):
        n_g = self.n_gaussians
        lat = self.latent_dim

        stride = n_g * lat

        mu = mdn_out[:, :, :stride]
        sigma = mdn_out[:, :, stride : 2 * stride]
        pi = mdn_out[:, :, 2 * stride : 2 * stride + n_g]

        pi = F.log_softmax(pi, dim=-1)  # log softmax like ctallec
        sigma = torch.exp(sigma)

        mu = mu.view(mu.size(0), mu.size(1), n_g, lat)
        sigma = sigma.view(sigma.size(0), sigma.size(1), n_g, lat)

        return pi, mu, sigma

    def death_loss(self, death_logits, target_death):
        death_logits = death_logits.squeeze(-1)
        return F.binary_cross_entropy_with_logits(death_logits, target_death)

    def sample(self, pi, mu, sigma):
        # pi is now log_softmax, need exp for multinomial
        batch_size, seq_len, n_gaussians = pi.shape
        pi_exp = torch.exp(pi)
        pi_flat = pi_exp.view(-1, n_gaussians)

        indices = torch.multinomial(pi_flat, 1)
        indices = indices.view(batch_size, seq_len)
        indices = indices.unsqueeze(-1).unsqueeze(-1)
        indices = indices.expand(batch_size, seq_len, 1, self.latent_dim)

        selected_mu = torch.gather(mu, 2, indices).squeeze(2)
        selected_sigma = torch.gather(sigma, 2, indices).squeeze(2)

        z = selected_mu + selected_sigma * torch.randn_like(selected_mu)
        return z

    def loss_function(self, pi, mu, sigma, target_z):
        # pi is log_softmax already
        target_z = target_z.unsqueeze(2)

        log_prob = (
            -0.5 * torch.log(torch.tensor(2 * 3.14159))
            - torch.log(sigma)
            - 0.5 * ((target_z - mu) / sigma) ** 2
        )

        log_prob = log_prob.sum(dim=-1)  # sum over latent dims
        log_weighted = log_prob + pi  # pi is already log
        log_likelihood = torch.logsumexp(log_weighted, dim=-1)

        return -log_likelihood.mean()


if __name__ == "__main__":
    model = MDNRNN(latent_dim=64, action_dim=3, hidden_dim=512)

    batch_size = 4
    seq_len = 10

    z = torch.randn(batch_size, seq_len, 64)
    action = torch.randint(0, 3, (batch_size, seq_len))
    target_z = torch.randn(batch_size, seq_len, 64)
    target_death = torch.zeros(batch_size, seq_len)
    target_death[:, -1] = 1.0

    mdn_out, death_logits, hidden = model(z, action)
    pi, mu, sigma = model.get_mdn_params(mdn_out)
    z_sample = model.sample(pi, mu, sigma)

    mdn_loss = model.loss_function(pi, mu, sigma, target_z)
    d_loss = model.death_loss(death_logits, target_death)

    total = sum(p.numel() for p in model.parameters())
    print(f"Params: {total:,}")
    print(f"Input z: {z.shape}")
    print(f"MDN out: {mdn_out.shape}")
    print(f"Death logits: {death_logits.shape}")
    print(f"pi: {pi.shape}, mu: {mu.shape}, sigma: {sigma.shape}")
    print(f"Sampled z: {z_sample.shape}")
    print(f"MDN loss: {mdn_loss.item():.2f}")
    print(f"Death loss: {d_loss.item():.2f}")
