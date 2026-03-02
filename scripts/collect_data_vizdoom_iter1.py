#!/usr/bin/env python3
"""
Collect VizDoom data using the trained z-only controller.
Mixes controller actions with random actions for exploration.

Usage:
    python collect_data_vizdoom_iter1.py --episodes 10000
"""

import sys

sys.path.append("..")

import os
import argparse
import time
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
        out = float((self.W2 @ hidden + self.b2)[0])
        if out < -0.33:
            return 0
        elif out > 0.33:
            return 2
        else:
            return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../outputs/vizdoom/iter0",
        help="Where to load VAE and controller from",
    )
    parser.add_argument("--output_dir", type=str, default="../outputs/vizdoom/iter1")
    parser.add_argument(
        "--random_fraction",
        type=float,
        default=0.2,
        help="Fraction of actions that are random for exploration",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cpu")

    # Load VAE for encoding frames to z
    vae = ConvVAE(latent_dim=64, kl_tolerance=0.5).to(device)
    vae.load_state_dict(
        torch.load(os.path.join(args.data_dir, "vae.pth"), map_location=device)
    )
    vae.eval()
    print("VAE loaded")

    # Load controller
    ctrl_path = os.path.join(args.data_dir, "controller_zonly_best.npy")
    if not os.path.exists(ctrl_path):
        ctrl_path = os.path.join(args.data_dir, "controller_zonly_params.npy")
    weights = np.load(ctrl_path)
    controller = ZOnlyController(weights)
    print(f"Controller loaded from {ctrl_path}")

    # Set up VizDoom
    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, "take_cover.cfg"))
    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_window_visible(False)
    game.init()

    actions_list = [[0, 0], [1, 0], [0, 1]]

    all_frames = []
    all_actions = []
    all_dones = []
    episode_lengths = []
    start_time = time.time()

    for episode in range(args.episodes):
        game.new_episode()
        episode_length = 0

        while not game.is_episode_finished():
            state = game.get_state()
            screen = state.screen_buffer
            frame = np.array(Image.fromarray(screen).resize((64, 64)))
            all_frames.append(frame)

            # Choose action: controller or random
            if np.random.rand() < args.random_fraction:
                action_idx = np.random.randint(3)
            else:
                frame_t = (
                    torch.from_numpy(frame).float().permute(2, 0, 1).unsqueeze(0)
                    / 255.0
                )
                with torch.no_grad():
                    mu, logvar = vae.encode(frame_t)
                    z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
                action_idx = controller.act(z[0].numpy())

            game.make_action(actions_list[action_idx])
            all_actions.append(action_idx)

            if game.is_episode_finished() and game.is_player_dead():
                all_dones.append(1.0)
            else:
                all_dones.append(0.0)

            episode_length += 1

        episode_lengths.append(episode_length)

        # Save chunks every 1000 episodes
        if (episode + 1) % 1000 == 0:
            chunk_num = (episode + 1) // 1000
            np.save(
                f"{args.output_dir}/frames_chunk{chunk_num}.npy",
                np.array(all_frames, dtype=np.uint8),
            )
            np.save(
                f"{args.output_dir}/actions_chunk{chunk_num}.npy",
                np.array(all_actions, dtype=np.int64),
            )
            np.save(
                f"{args.output_dir}/dones_chunk{chunk_num}.npy",
                np.array(all_dones, dtype=np.float32),
            )

            elapsed = time.time() - start_time
            avg_len = np.mean(episode_lengths)
            fps = sum(episode_lengths) / elapsed
            print(
                f"Episode {episode+1}/{args.episodes} | "
                f"Avg length: {avg_len:.0f} | "
                f"Chunk {chunk_num} saved | {fps:.0f} FPS"
            )

            all_frames.clear()
            all_actions.clear()
            all_dones.clear()

    game.close()

    # Save remaining
    if all_frames:
        chunk_num = (args.episodes // 1000) + 1
        np.save(
            f"{args.output_dir}/frames_chunk{chunk_num}.npy",
            np.array(all_frames, dtype=np.uint8),
        )
        np.save(
            f"{args.output_dir}/actions_chunk{chunk_num}.npy",
            np.array(all_actions, dtype=np.int64),
        )
        np.save(
            f"{args.output_dir}/dones_chunk{chunk_num}.npy",
            np.array(all_dones, dtype=np.float32),
        )

    np.save(f"{args.output_dir}/episode_lengths.npy", np.array(episode_lengths))

    print(f"\nCollection complete!")
    print(f"Episodes: {len(episode_lengths)}")
    print(f"Avg length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Random fraction: {args.random_fraction}")
    print(f"Output: {args.output_dir}")
