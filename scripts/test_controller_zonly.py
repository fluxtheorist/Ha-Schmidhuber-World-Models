#!/usr/bin/env python3
"""Test z-only controller in real VizDoom."""

import sys

sys.path.append("..")

import os
import argparse
import numpy as np
import torch
import vizdoom as vzd
from PIL import Image
from models.vae import ConvVAE


class ZOnlyController:
    def __init__(self, weights):
        self.params = np.asarray(weights, dtype=np.float32).flatten()
        z_dim, hidden_dim = 64, 40
        idx = 0
        self.W1 = self.params[idx : idx + z_dim * hidden_dim].reshape(hidden_dim, z_dim)
        idx += z_dim * hidden_dim
        self.b1 = self.params[idx : idx + hidden_dim]
        idx += hidden_dim
        self.W2 = self.params[idx : idx + hidden_dim].reshape(1, hidden_dim)
        idx += hidden_dim
        self.b2 = self.params[idx : idx + 1]

    def act(self, z):
        hidden = np.tanh(self.W1 @ z + self.b1)
        out = float(self.W2 @ hidden + self.b2)
        if out < -0.33:
            return 0  # stay
        elif out > 0.33:
            return 2  # right
        else:
            return 1  # left


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../outputs/vizdoom/iter0")
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cpu")

    vae = ConvVAE(latent_dim=64, kl_tolerance=0.5).to(device)
    vae.load_state_dict(
        torch.load(os.path.join(args.data_dir, "vae.pth"), map_location=device)
    )
    vae.eval()

    ctrl_path = os.path.join(args.data_dir, "controller_zonly_best.npy")
    if not os.path.exists(ctrl_path):
        ctrl_path = os.path.join(args.data_dir, "controller_zonly_params.npy")
    weights = np.load(ctrl_path)
    controller = ZOnlyController(weights)
    print(f"Controller loaded from {ctrl_path}")

    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, "take_cover.cfg"))
    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_window_visible(False)
    game.init()

    actions_list = [[0, 0], [1, 0], [0, 1]]

    print(f"\nRunning {args.episodes} episodes...")
    all_steps = []

    for ep in range(args.episodes):
        game.new_episode()
        steps = 0

        while not game.is_episode_finished():
            screen = game.get_state().screen_buffer
            frame = np.array(Image.fromarray(screen).resize((64, 64)))
            frame_t = (
                torch.from_numpy(frame).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            )

            with torch.no_grad():
                mu, logvar = vae.encode(frame_t)
                z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

            z_np = z[0].numpy()
            action_idx = controller.act(z_np)
            game.make_action(actions_list[action_idx])
            steps += 1

        all_steps.append(steps)
        if (ep + 1) % 10 == 0 or ep < 5:
            print(f"  Episode {ep+1:3d}: {steps} steps")

    game.close()

    print(f"\n{'='*40}")
    print(f"Results over {args.episodes} episodes:")
    print(f"  Steps: {np.mean(all_steps):.1f} ± {np.std(all_steps):.1f}")
    print(f"  Best:  {max(all_steps)}")
    print(f"  Worst: {min(all_steps)}")
    print(f"  Median: {np.median(all_steps):.0f}")
    print(f"\n  Ha's z-only+hidden: 788 ± 141")
    print(f"  Random policy:      ~246")
    print(f"  Solve threshold:    750")
