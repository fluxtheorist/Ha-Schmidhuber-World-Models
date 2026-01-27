"""
Train controller using iteration 1 VAE and MDN-RNN
"""

import torch
import numpy as np
import cma
import gymnasium as gym
from PIL import Image
from vae import ConvVAE
from mdn_rnn import MDNRNN
from controller import Controller

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load ITERATION 1 models
vae = ConvVAE(latent_dim=32)
vae.load_state_dict(torch.load("outputs/vae_iter1.pth"))
vae.to(device)
vae.eval()

mdn_rnn = MDNRNN(latent_dim=32, action_dim=3, hidden_dim=256, n_gaussians=5)
mdn_rnn.load_state_dict(torch.load("outputs/mdn_rnn_iter1.pth"))
mdn_rnn.to(device)
mdn_rnn.eval()

controller = Controller()
controller.to(device)


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


def rollout():
    env = gym.make("CarRacing-v3", continuous=True)
    obs, _ = env.reset()
    obs = np.array(Image.fromarray(obs).resize((64, 64)))

    hidden = (torch.zeros(1, 1, 256).to(device), torch.zeros(1, 1, 256).to(device))
    total_reward = 0

    for step in range(500):
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float() / 255.0
            obs_t = obs_t.permute(2, 0, 1).unsqueeze(0).to(device)
            mu, _ = vae.encode(obs_t)
            z = mu.squeeze(0)
            h = hidden[0].squeeze(0).squeeze(0)

            action = controller(z, h).cpu().numpy()

        action_env = np.array([action[0], (action[1] + 1) / 2, (action[2] + 1) / 2])
        obs, reward, terminated, truncated, _ = env.step(action_env)
        obs = np.array(Image.fromarray(obs).resize((64, 64)))
        total_reward += reward

        with torch.no_grad():
            z_input = z.unsqueeze(0).unsqueeze(0)
            a_input = (
                torch.from_numpy(action).float().unsqueeze(0).unsqueeze(0).to(device)
            )
            _, hidden = mdn_rnn(z_input, a_input, hidden)

        if terminated or truncated:
            break

    env.close()
    return total_reward


if __name__ == "__main__":
    n_params = sum(p.numel() for p in controller.parameters())
    print(f"Optimizing {n_params} parameters")

    es = cma.CMAEvolutionStrategy(n_params * [0], 0.5)

    for generation in range(50):
        solutions = es.ask()

        fitness = []
        for params in solutions:
            set_controller_params(np.array(params))
            reward = rollout()
            fitness.append(-reward)

        es.tell(solutions, fitness)

        best_reward = -min(fitness)
        mean_reward = -np.mean(fitness)
        print(f"Gen {generation+1}: Best={best_reward:.1f}, Mean={mean_reward:.1f}")

    # Save
    best_params = np.array(es.result.xbest)
    np.save("outputs/controller_params_iter1.npy", best_params)
    print("Saved to outputs/controller_params_iter1.npy")

    # Verify
    set_controller_params(best_params)
    rewards = [rollout() for _ in range(5)]
    print(f"Verification: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
