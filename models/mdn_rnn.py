"""
MDN-RNN matching Ha's doomrnn.py architecture exactly.

Input per timestep: [z(64), action_scalar(1), restart(1)] = 66 dims
LSTM hidden: 512
Output: restart_logit(1) + GMM params(64*5*3 = 960) = 961

Used for both training (forward_train) and inference (forward_step).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


RESTART_FACTOR = 10.0


class MDNRNN(nn.Module):
    def __init__(self, z_size=64, n_mix=5, rnn_size=512):
        super().__init__()
        self.z_size = z_size
        self.n_mix = n_mix
        self.rnn_size = rnn_size
        self.input_size = z_size + 1 + 1  # z + action_scalar + restart

        self.lstm = nn.LSTMCell(self.input_size, rnn_size)

        self.nout = z_size * n_mix * 3 + 1  # 961
        self.output_w = nn.Linear(rnn_size, self.nout)

    # ================================================================
    # Training
    # ================================================================

    def forward_train(self, batch_z, batch_action, batch_restart, initial_state=None):
        """Training forward pass. Matches Ha's custom_rnn_autodecoder.

        Args:
            batch_z: (batch, seq_len, 64) - full sequence
            batch_action: (batch, seq_len) - scalar actions as float
            batch_restart: (batch, seq_len) - restart flags
            initial_state: (h, c) each (batch, rnn_size), or None

        Returns:
            z_cost, r_cost, (h_detached, c_detached)
        """
        batch_size = batch_z.shape[0]
        LENGTH = batch_z.shape[1] - 1
        device = batch_z.device

        input_z = batch_z[:, :LENGTH, :]
        input_action = batch_action[:, :LENGTH]
        input_restart = batch_restart[:, :LENGTH]
        target_z = batch_z[:, 1:, :]
        target_restart = batch_restart[:, 1:]

        input_seq = torch.cat(
            [
                input_z,
                input_action.unsqueeze(-1),
                input_restart.unsqueeze(-1),
            ],
            dim=2,
        )

        if initial_state is None:
            h = torch.zeros(batch_size, self.rnn_size, device=device)
            c = torch.zeros(batch_size, self.rnn_size, device=device)
        else:
            h, c = initial_state

        zero_h = torch.zeros_like(h)
        zero_c = torch.zeros_like(c)

        outputs = []
        for i in range(LENGTH):
            inp = input_seq[:, i, :]
            restart_flag = input_restart[:, i] > 0.5
            if restart_flag.any():
                mask = restart_flag.unsqueeze(1)
                c = torch.where(mask, zero_c, c)
                h = torch.where(mask, zero_h, h)
            h, c = self.lstm(inp, (h, c))
            outputs.append(h)

        rnn_output = torch.stack(outputs, dim=1)
        flat_output = self.output_w(rnn_output.reshape(-1, self.rnn_size))

        out_restart_logits = flat_output[:, 0]
        gmm_output = flat_output[:, 1:].reshape(-1, self.n_mix * 3)

        logmix, mean, logstd = torch.chunk(gmm_output, 3, dim=1)
        logmix = logmix - torch.logsumexp(logmix, dim=1, keepdim=True)

        flat_target = target_z.reshape(-1, 1)
        log_sqrt_2pi = np.log(np.sqrt(2.0 * np.pi))
        log_prob = (
            -0.5 * ((flat_target - mean) / torch.exp(logstd)) ** 2
            - logstd
            - log_sqrt_2pi
        )
        v = torch.logsumexp(logmix + log_prob, dim=1, keepdim=True)
        z_cost = -torch.mean(v)

        flat_target_restart = target_restart.reshape(-1, 1)
        r_cost = F.binary_cross_entropy_with_logits(
            out_restart_logits.reshape(-1, 1), flat_target_restart, reduction="none"
        )
        factor = 1.0 + flat_target_restart * (RESTART_FACTOR - 1.0)
        r_cost = torch.mean(factor * r_cost)

        return z_cost, r_cost, (h.detach(), c.detach())

    # ================================================================
    # Inference (single step, for dream environment)
    # ================================================================

    def forward_step(self, z, action, restart, state):
        """Single-step forward for dream rollout. Matches Ha's _step.

        Args:
            z: (1, z_size) current latent
            action: scalar float (0, 1, or 2)
            restart: scalar float (0 or 1)
            state: (h, c) each (1, rnn_size)

        Returns:
            logmix: (z_size, n_mix)
            mean: (z_size, n_mix)
            logstd: (z_size, n_mix)
            restart_logit: scalar float
            new_state: (h, c)
        """
        h, c = state
        device = z.device

        if restart > 0.5:
            h = torch.zeros_like(h)
            c = torch.zeros_like(c)

        inp = torch.cat(
            [
                z,
                torch.tensor([[action]], device=device, dtype=torch.float32),
                torch.tensor([[restart]], device=device, dtype=torch.float32),
            ],
            dim=1,
        )

        h, c = self.lstm(inp, (h, c))
        output = self.output_w(h)

        restart_logit = output[0, 0].item()
        gmm_out = output[0, 1:].reshape(self.z_size, self.n_mix * 3)

        logmix, mean, logstd = torch.chunk(gmm_out, 3, dim=1)
        logmix = logmix - torch.logsumexp(logmix, dim=1, keepdim=True)

        return logmix, mean, logstd, restart_logit, (h, c)

    def sample_z(self, logmix, mean, logstd, temperature=1.25):
        """Sample next z from GMM. Matches Ha's dream env sampling.

        Args:
            logmix: (z_size, n_mix)
            mean: (z_size, n_mix)
            logstd: (z_size, n_mix)
            temperature: float

        Returns:
            z: (z_size,) numpy array
        """
        logmix_np = logmix.detach().cpu().numpy()
        mean_np = mean.detach().cpu().numpy()
        logstd_np = logstd.detach().cpu().numpy()

        logmix2 = logmix_np / temperature
        logmix2 -= logmix2.max(axis=1, keepdims=True)
        logmix2 = np.exp(logmix2)
        logmix2 /= logmix2.sum(axis=1, keepdims=True)

        z_size = logmix_np.shape[0]
        chosen_mean = np.zeros(z_size)
        chosen_logstd = np.zeros(z_size)
        for j in range(z_size):
            idx = np.random.choice(self.n_mix, p=logmix2[j])
            chosen_mean[j] = mean_np[j, idx]
            chosen_logstd[j] = logstd_np[j, idx]

        rand_gaussian = np.random.randn(z_size) * np.sqrt(temperature)
        next_z = chosen_mean + np.exp(chosen_logstd) * rand_gaussian
        return next_z

    def init_state(self, batch_size, device):
        h = torch.zeros(batch_size, self.rnn_size, device=device)
        c = torch.zeros(batch_size, self.rnn_size, device=device)
        return (h, c)


if __name__ == "__main__":
    model = MDNRNN(z_size=64, n_mix=5, rnn_size=512)
    total = sum(p.numel() for p in model.parameters())
    print(f"Params: {total:,}")

    z = torch.randn(1, 64)
    state = model.init_state(1, torch.device("cpu"))
    logmix, mean, logstd, restart_logit, state = model.forward_step(z, 0.0, 1.0, state)
    next_z = model.sample_z(logmix, mean, logstd, temperature=1.25)
    print(f"logmix: {logmix.shape}, mean: {mean.shape}, logstd: {logstd.shape}")
    print(f"restart_logit: {restart_logit:.4f}")
    print(f"next_z: shape={next_z.shape}, mean={next_z.mean():.4f}")
