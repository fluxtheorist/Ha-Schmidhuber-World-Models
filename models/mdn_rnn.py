import torch
import torch.nn as nn
import torch.nn.functional as F


class MDNRNN(nn.Module):
    def __init__(self, latent_dim=32, action_dim=3, hidden_dim=256, n_gaussians=5):
        super().__init__()

        # Input: z + action + restart flag (matches Ha's code)
        input_dim = latent_dim + action_dim + 1

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.n_gaussians = n_gaussians

        # Output: GMM params + restart logit
        # Ha: NOUT = WIDTH * KMIX * 3 + 1
        output_dim = latent_dim * n_gaussians * 3 + 1
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z, action, restart, hidden=None):
        """Multi-step forward with restart-based hidden state reset.

        Uses nn.LSTM for efficient gradient flow. Processes segments
        between restart boundaries, resetting hidden state at restarts.
        """
        batch_size, seq_len, _ = z.shape
        device = z.device

        # One-hot encode actions
        action_oh = F.one_hot(action.long(), num_classes=self.action_dim).float()

        # Concatenate [z, action_oh, restart]
        x = torch.cat([z, action_oh, restart.unsqueeze(-1)], dim=-1)

        if hidden is None:
            h = torch.zeros(1, batch_size, self.hidden_dim, device=device)
            c = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        else:
            h, c = hidden

        # Find timesteps where ANY sample has restart=1
        restart_any = restart.sum(dim=0) > 0  # (seq_len,)
        restart_times = torch.where(restart_any)[0].tolist()

        # Process segments between restart points using nn.LSTM
        outputs = []
        seg_start = 0

        for rt in restart_times:
            # Process segment before this restart
            if rt > seg_start:
                seg_out, (h, c) = self.lstm(x[:, seg_start:rt, :], (h, c))
                outputs.append(seg_out)

            # Reset hidden for samples with restart=1 at this timestep
            restart_mask = restart[:, rt].bool()
            if restart_mask.any():
                mask = restart_mask.unsqueeze(0).unsqueeze(-1)  # (1, batch, 1)
                h = torch.where(mask, torch.zeros_like(h), h)
                c = torch.where(mask, torch.zeros_like(c), c)

            # Process the restart timestep
            seg_out, (h, c) = self.lstm(x[:, rt : rt + 1, :], (h, c))
            outputs.append(seg_out)
            seg_start = rt + 1

        # Process remaining segment after last restart
        if seg_start < seq_len:
            seg_out, (h, c) = self.lstm(x[:, seg_start:, :], (h, c))
            outputs.append(seg_out)

        # Concatenate all segments
        lstm_out = torch.cat(outputs, dim=1)  # (batch, seq_len, hidden_dim)

        # Output projection
        out = self.fc_out(lstm_out)

        return out, (h, c)

    def get_mdn_params(self, output):
        """Extract MDN parameters from output.

        Ha's layout: output[:, 0] = restart_logit, output[:, 1:] = GMM
        GMM is reshaped to (latent_dim, KMIX*3) then split into logmix, mean, logstd
        """
        restart_logits = output[:, :, 0]  # (batch, seq_len)
        gmm_out = output[:, :, 1:]  # (batch, seq_len, latent*gauss*3)

        n_g = self.n_gaussians
        lat = self.latent_dim

        # Reshape to (batch, seq_len, latent_dim, n_gaussians * 3)
        gmm_out = gmm_out.view(gmm_out.size(0), gmm_out.size(1), lat, n_g * 3)

        # Split into logmix, mean, logstd (matching Ha's get_mdn_coef)
        logmix = gmm_out[:, :, :, :n_g]
        mean = gmm_out[:, :, :, n_g : 2 * n_g]
        logstd = gmm_out[:, :, :, 2 * n_g :]

        # Normalize logmix
        logmix = logmix - torch.logsumexp(logmix, dim=-1, keepdim=True)

        return restart_logits, logmix, mean, logstd

    def loss_function(self, logmix, mean, logstd, target_z):
        """GMM loss matching Ha's get_lossfunc.

        target_z: (batch, seq_len, latent_dim)
        logmix, mean, logstd: (batch, seq_len, latent_dim, n_gaussians)
        """
        target = target_z.unsqueeze(-1)  # (batch, seq_len, latent_dim, 1)

        log_sqrt_2pi = 0.5 * torch.log(
            torch.tensor(2.0 * 3.14159265, device=mean.device)
        )
        log_prob = (
            -0.5 * ((target - mean) / torch.exp(logstd)) ** 2 - logstd - log_sqrt_2pi
        )

        # Add logmix and logsumexp over gaussians
        v = logmix + log_prob
        v = torch.logsumexp(v, dim=-1)  # (batch, seq_len, latent_dim)

        return -v.mean()

    def restart_loss(self, restart_logits, target_restart, restart_factor=10.0):
        """Restart loss with weighting, matching Ha's implementation."""
        loss = F.binary_cross_entropy_with_logits(
            restart_logits, target_restart, reduction="none"
        )
        weight = 1.0 + target_restart * (restart_factor - 1.0)
        return (loss * weight).mean()

    def sample(self, logmix, mean, logstd, temperature=1.0):
        """Sample next z from GMM with temperature."""
        logmix2 = logmix / temperature
        logmix2 = logmix2 - torch.logsumexp(logmix2, dim=-1, keepdim=True)
        probs = torch.exp(logmix2)

        batch_size = probs.size(0)
        lat = self.latent_dim
        n_g = self.n_gaussians

        probs_flat = probs.view(-1, n_g)
        indices = torch.multinomial(probs_flat, 1)
        indices = indices.view(batch_size, lat, 1)

        chosen_mean = torch.gather(mean, -1, indices).squeeze(-1)
        chosen_logstd = torch.gather(logstd, -1, indices).squeeze(-1)

        z = chosen_mean + torch.exp(chosen_logstd) * torch.randn_like(
            chosen_mean
        ) * torch.sqrt(torch.tensor(temperature))
        return z

    def init_hidden(self, batch_size, device):
        """Initialize hidden state to zeros."""
        h = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        return (h, c)


if __name__ == "__main__":
    model = MDNRNN(latent_dim=64, action_dim=3, hidden_dim=512, n_gaussians=5)

    batch_size = 4
    seq_len = 10

    z = torch.randn(batch_size, seq_len, 64)
    action = torch.randint(0, 3, (batch_size, seq_len))
    restart = torch.zeros(batch_size, seq_len)
    restart[:, 0] = 1.0

    output, hidden = model(z, action, restart)
    restart_logits, logmix, mean, logstd = model.get_mdn_params(output)

    target_z = torch.randn(batch_size, seq_len, 64)
    target_restart = torch.zeros(batch_size, seq_len)
    target_restart[:, -1] = 1.0

    z_loss = model.loss_function(logmix, mean, logstd, target_z)
    r_loss = model.restart_loss(restart_logits, target_restart)

    total = sum(p.numel() for p in model.parameters())
    print(f"Params: {total:,}")
    print(f"Output shape: {output.shape}")
    print(f"restart_logits: {restart_logits.shape}")
    print(f"logmix: {logmix.shape}, mean: {mean.shape}, logstd: {logstd.shape}")
    print(f"Z loss: {z_loss.item():.2f}")
    print(f"Restart loss: {r_loss.item():.2f}")
    print(f"Total loss: {(z_loss + r_loss).item():.2f}")
