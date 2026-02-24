#!/usr/bin/env python3
"""
Test dream-trained controller in real VizDoom Take Cover.

The controller was trained entirely in the MDN-RNN dream environment.
This script tests transfer to the real game.

Usage:
    python test_controller_vizdoom.py --data_dir ../outputs/vizdoom/iter0 --episodes 100
"""

import sys

sys.path.append("..")

import os
import argparse
import numpy as np
import torch
import vizdoom as vzd
from PIL import Image

from models.vae import ConvVAE
from models.mdn_rnn import MDNRNN


class Controller:
    """Same controller as training — linear map [z, c, h] -> action."""

    def __init__(self, weights):
        self.weights = np.asarray(weights, dtype=np.float32).flatten()

    def act(self, z, h, c):
        obs = np.concatenate([z, c, h])
        out = float(np.dot(self.weights, obs))
        if out < -0.33:
            return 0  # move left
        elif out > 0.33:
            return 2  # move right
        else:
            return 1  # stay


def run_episode(game, vae, rnn, controller, device, render=False):
    """Run one episode in real VizDoom.

    Pipeline per step:
    1. Get screen from VizDoom
    2. Encode with VAE -> z
    3. Controller([z, c, h]) -> action
    4. Execute action in VizDoom
    5. RNN step to update hidden state
    """
    game.new_episode()

    # VizDoom actions: [move_left, move_right]
    actions = [[0, 0], [1, 0], [0, 1]]

    # Initialize RNN state
    state = rnn.init_state(1, device)
    restart = 1.0  # First step is a restart
    prev_z = None

    total_reward = 0
    steps = 0

    while not game.is_episode_finished():
        # Get frame and encode
        screen = game.get_state().screen_buffer
        frame = np.array(Image.fromarray(screen).resize((64, 64)))
        frame_t = (
            torch.from_numpy(frame).float().permute(2, 0, 1).unsqueeze(0).to(device)
            / 255.0
        )

        with torch.no_grad():
            mu, logvar = vae.encode(frame_t)
            z = mu  # Use deterministic encoding for real-world testing

        z_np = z[0].cpu().numpy()
        h_np = state[0][0].cpu().numpy()
        c_np = state[1][0].cpu().numpy()

        # Controller action
        action_idx = controller.act(z_np, h_np, c_np)
        action_float = float(action_idx)

        # Execute in VizDoom
        reward = game.make_action(actions[action_idx])
        total_reward += reward
        steps += 1

        # Update RNN state (feed current z and action)
        with torch.no_grad():
            _, _, _, restart_logit, state = rnn.forward_step(
                z, action_float, restart, state
            )

        # Update restart flag based on RNN prediction (not used for game termination,
        # but needed to maintain consistent RNN state)
        restart = 1.0 if restart_logit > 0 else 0.0

    return total_reward, steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../outputs/vizdoom/iter0")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu")  # CPU for inference

    # Load VAE
    vae_path = os.path.join(args.data_dir, "vae.pth")
    vae = ConvVAE(latent_dim=64, kl_tolerance=0.5).to(device)
    vae.load_state_dict(torch.load(vae_path, map_location=device))
    vae.eval()
    print(f"VAE loaded from {vae_path}")

    # Load MDN-RNN
    rnn_path = os.path.join(args.data_dir, "mdn_rnn.pth")
    rnn = MDNRNN(z_size=64, n_mix=5, rnn_size=512).to(device)
    rnn.load_state_dict(torch.load(rnn_path, map_location=device))
    rnn.eval()
    print(f"MDN-RNN loaded from {rnn_path}")

    # Load controller
    ctrl_path = os.path.join(args.data_dir, "controller_best.npy")
    if not os.path.exists(ctrl_path):
        ctrl_path = os.path.join(args.data_dir, "controller_params.npy")
    weights = np.load(ctrl_path)
    controller = Controller(weights)
    print(f"Controller loaded from {ctrl_path}")
    print(f"  Weight norm: {np.linalg.norm(weights):.4f}")
    print(f"  Weight range: [{weights.min():.4f}, {weights.max():.4f}]")

    # Set up VizDoom
    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, "take_cover.cfg"))
    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_window_visible(args.render)
    game.init()

    print(f"\nRunning {args.episodes} episodes in real VizDoom...")
    print(f"{'Episode':>8} {'Steps':>8} {'Reward':>8}")
    print("-" * 28)

    all_rewards = []
    all_steps = []

    for ep in range(args.episodes):
        reward, steps = run_episode(game, vae, rnn, controller, device, args.render)
        all_rewards.append(reward)
        all_steps.append(steps)

        if (ep + 1) % 10 == 0 or ep < 5:
            print(f"{ep+1:>8} {steps:>8} {reward:>8.1f}")

    game.close()

    # Summary
    print(f"\n{'='*40}")
    print(f"Results over {args.episodes} episodes:")
    print(f"  Reward: {np.mean(all_rewards):.1f} ± {np.std(all_rewards):.1f}")
    print(f"  Steps:  {np.mean(all_steps):.1f} ± {np.std(all_steps):.1f}")
    print(f"  Best:   {max(all_steps)} steps ({max(all_rewards):.1f} reward)")
    print(f"  Worst:  {min(all_steps)} steps ({min(all_rewards):.1f} reward)")
    print(f"  Median: {np.median(all_steps):.0f} steps")

    # Ha's paper reports ~750 average for dream-trained controller
    print(f"\n  Ha's paper baseline: ~750 avg steps")
    print(f"  Random policy:       ~210 avg steps")
