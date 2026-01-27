import torch
import numpy as np
import cma
from vae import ConvVAE
from mdn_rnn import MDNRNN
from controller import Controller
from reward_predictor import RewardPredictor
from dream_env import DreamEnv

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load models
vae = ConvVAE(latent_dim=32)
vae.load_state_dict(torch.load("outputs/vae.pth"))
vae.to(device)
vae.eval()

mdn_rnn = MDNRNN(latent_dim=32, action_dim=3, hidden_dim=256, n_gaussians=5)
mdn_rnn.load_state_dict(torch.load("outputs/mdn_rnn.pth"))
mdn_rnn.to(device)
mdn_rnn.eval()

reward_predictor = RewardPredictor(latent_dim=32, hidden_dim=64)
reward_predictor.load_state_dict(torch.load("outputs/reward_predictor.pth"))
reward_predictor.to(device)
reward_predictor.eval()

controller = Controller()
controller.to(device)

# Get initial z vectors
frames = np.load("outputs/frames.npy")
frames_tensor = torch.from_numpy(frames).float() / 255.0
frames_tensor = frames_tensor.permute(0, 3, 1, 2)

with torch.no_grad():
    all_z = []
    for i in range(0, len(frames_tensor), 256):
        batch = frames_tensor[i : i + 256].to(device)
        mu, _ = vae.encode(batch)
        all_z.append(mu.cpu())
    all_z = torch.cat(all_z, dim=0)

print(f"Loaded {len(all_z)} initial z vectors")

# Create dream environment
dream = DreamEnv(mdn_rnn, reward_predictor, all_z, device, temperature=1.0)


def set_params(controller, params):
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


def dream_rollout(controller, dream):
    z, h = dream.reset()
    total_reward = 0

    for step in range(500):
        action = controller(z, h)
        z, h, reward, done = dream.step(action)
        total_reward += reward
        if done:
            break

    return total_reward


# CMA-ES in dreams
n_params = sum(p.numel() for p in controller.parameters())
print(f"Optimizing {n_params} parameters in dreams")

es = cma.CMAEvolutionStrategy(n_params * [0], 0.5, {"popsize": 16})

for gen in range(100):
    solutions = es.ask()

    rewards = []
    for params in solutions:
        set_params(controller, np.array(params))
        r = dream_rollout(controller, dream)
        rewards.append(r)

    es.tell(solutions, [-r for r in rewards])

    print(f"Gen {gen+1}: Best={max(rewards):.1f}, Mean={np.mean(rewards):.1f}")

# Save dream-trained controller
set_params(controller, es.result.xbest)
torch.save(controller.state_dict(), "outputs/controller_dream.pth")
print("Saved dream-trained controller")
