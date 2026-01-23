import torch


class DreamEnv:
    # Enviornment that runs in the MDN-RNN imagination

    def __init__(self, mdn_rnn, initial_z_bank, device, temperature=1.0):
        self.mdn_rnn = mdn_rnn
        self.device = device
        self.temperature = temperature
        self.initial_z_bank = initial_z_bank

    def reset(self):
        # Start dream episode

        # Pick random starting z
        idx = torch.randint(len(self.initial_z_bank), (1,)).item()
        self.z = self.initial_z_bank[idx].clone().to(self.device)

        # Reset hidden state
        self.hidden = (
            torch.zeros(1, 1, 256).to(self.device),
            torch.zeros(1, 1, 256).to(self.device),
        )
        self.steps = 0
        h = self.hidden[0].squeeze(0).squeeze(0)
        return self.z, h

    def step(self, action):
        # Take action to get next state from MDN-RNN imagination
        with torch.no_grad():
            z_input = self.z.unsqueeze(0).unsqueeze(0)
            a_input = action.unsqueeze(0).unsqueeze(0)

            # MDN-RNN predicts next z
            mdn_out, self.hidden = self.mdn_rnn(z_input, a_input, self.hidden)
            pi, mu, sigma = self.mdn_rnn.get_mdn_params(mdn_out)

            # Apply temp
            sigma = sigma * self.temperature

            # Sample next z
            self.z = self.mdn_rnn.sample(pi, mu, sigma).squeeze(0).squeeze(0)

        self.steps += 1
        h = self.hidden[0].squeeze(0).squeeze(0)

        done = self.steps >= 500
        reward = 0.1

        return self.z, h, reward, done
