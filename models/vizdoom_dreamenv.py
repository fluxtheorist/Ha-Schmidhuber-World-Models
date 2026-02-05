import torch
import torch.nn.functional as F


class DreamEnvVizDoom:
    def __init__(
        self, mdn_rnn, initial_z_bank, device, temperature=1.15, max_steps=2100
    ):
        self.mdn_rnn = mdn_rnn
        self.device = device
        self.temperature = temperature
        self.max_steps = max_steps
        self.initial_z_bank = initial_z_bank

    def reset(self):
        idx = torch.randint(len(self.initial_z_bank), (1,)).item()
        self.z = self.initial_z_bank[idx].clone().to(self.device)
        hidden_dim = self.mdn_rnn.hidden_dim
        self.hidden = (
            torch.zeros(1, 1, hidden_dim).to(self.device),
            torch.zeros(1, 1, hidden_dim).to(self.device),
        )
        self.steps = 0
        h = self.hidden[0].squeeze(0).squeeze(0)
        c = self.hidden[1].squeeze(0).squeeze(0)
        return self.z, h, c

    def step(self, action_idx):
        with torch.no_grad():
            z_input = self.z.unsqueeze(0).unsqueeze(0)
            a_input = torch.tensor(action_idx, dtype=torch.long).to(self.device)
            mdn_output, death_logits, self.hidden = self.mdn_rnn(
                z_input, a_input, self.hidden
            )
            pi, mu, sigma = self.mdn_rnn.mdn.get_params(mdn_output)
            sigma = sigma * self.temperature
            self.z = self.mdn_rnn.sample(pi, mu, sigma).squeeze(0).squeeze(0)
            death_prob = torch.sigmoid(death_logits).squeeze().item()
            dead = death_prob > 0.5

        self.steps += 1
        reward = 1.0
        done = dead or (self.steps >= self.max_steps)
        h = self.hidden[0].squeeze(0).squeeze(0)
        c = self.hidden[1].squeeze(0).squeeze(0)
        return self.z, h, c, reward, done
