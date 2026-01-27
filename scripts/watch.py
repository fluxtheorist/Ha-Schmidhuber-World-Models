import sys

sys.path.append("..")

import torch
import numpy as np
import gymnasium as gym
import argparse
from PIL import Image
from models.vae import ConvVAE
from models.mdn_rnn import MDNRNN
from models.controller import Controller

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def set_controller_params(controller, params):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=0, help="Which iteration to watch")
    parser.add_argument(
        "--episodes", type=int, default=1, help="Number of episodes to run"
    )
    args = parser.parse_args()

    load_dir = f"../outputs/iter{args.iter}"
    print(f"Loading models from {load_dir}")

    vae = ConvVAE(latent_dim=32)
    vae.load_state_dict(torch.load(f"{load_dir}/vae.pth"))
    vae.to(device)
    vae.eval()

    mdn_rnn = MDNRNN(latent_dim=32, action_dim=3, hidden_dim=256, n_gaussians=5)
    mdn_rnn.load_state_dict(torch.load(f"{load_dir}/mdn_rnn.pth"))
    mdn_rnn.to(device)
    mdn_rnn.eval()

    controller = Controller()
    controller.to(device)

    params = np.load(f"{load_dir}/controller_params.npy")
    set_controller_params(controller, params)
    controller.eval()

    print("Models loaded! Starting...")

    env = gym.make("CarRacing-v3", continuous=True, render_mode="human")

    for episode in range(args.episodes):
        obs, _ = env.reset()
        obs = np.array(Image.fromarray(obs).resize((64, 64)))

        hidden = (torch.zeros(1, 1, 256).to(device), torch.zeros(1, 1, 256).to(device))
        total_reward = 0

        for step in range(1000):
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).float() / 255.0
                obs_tensor = obs_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
                mu, _ = vae.encode(obs_tensor)
                z = mu.squeeze(0)

                h = hidden[0].squeeze(0).squeeze(0)
                action = controller(z, h).cpu().numpy()

                action_env = np.array(
                    [action[0], (action[1] + 1) / 2, (action[2] + 1) / 2]
                )

                obs, reward, terminated, truncated, _ = env.step(action_env)
                obs = np.array(Image.fromarray(obs).resize((64, 64)))
                total_reward += reward

                z_input = z.unsqueeze(0).unsqueeze(0)
                a_tensor = torch.from_numpy(action).float().to(device)
                a_input = a_tensor.unsqueeze(0).unsqueeze(0)
                _, hidden = mdn_rnn(z_input, a_input, hidden)

            if terminated or truncated:
                break

        print(f"Episode {episode+1}: {total_reward:.1f}")

    env.close()
