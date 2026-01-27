import numpy as np
import torch
from controller import Controller
from vae import ConvVAE
from mdn_rnn import MDNRNN

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load models
vae = ConvVAE(latent_dim=32)
vae.load_state_dict(torch.load("outputs/vae.pth"))
vae.to(device)
vae.eval()

mdn_rnn = MDNRNN(latent_dim=32, action_dim=3, hidden_dim=256, n_gaussians=5)
mdn_rnn.load_state_dict(torch.load("outputs/mdn_rnn.pth"))
mdn_rnn.to(device)
mdn_rnn.eval()

controller = Controller()
controller.to(device)


# Load params and set
def set_controller_params(params):
    idx = 0
    for p in controller.parameters():
        size = p.numel()
        p.data = (
            torch.from_numpy(params[idx : idx + size])
            .float()
            .reshape(p.shape)
            .to(device)
        )
        idx += size


params = np.load("outputs/controller_params.npy")
set_controller_params(params)

# Import rollout function or define it, then test multiple times
from train_controller import rollout

rewards = []
for i in range(10):
    r = rollout(controller, vae, mdn_rnn)
    rewards.append(r)
    print(f"Run {i+1}: {r:.1f}")

print(f"\nAverage: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
