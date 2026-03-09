#!/usr/bin/env python3
"""
Test controller in real VizDoom, matching Ha's doomreal.py exactly.

Key: controller outputs continuous action which is fed to BOTH:
  1. RNN (as continuous float)
  2. VizDoom (thresholded to buttons)

This matches Ha's real-world wrapper where the same action value
goes to the RNN state update AND the game environment.

Usage:
    python test_controller_ha.py --data_dir ../outputs/vizdoom/ha_exact --episodes 100
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


def process_frame(screen_buffer):
    """Ha's _process_frame: crop, resize, invert."""
    crop_height = int(screen_buffer.shape[0] * 400 / 480)
    cropped = screen_buffer[:crop_height, :, :]
    img = np.array(Image.fromarray(cropped).resize((64, 64)))
    img_float = img.astype(np.float32) / 255.0
    img_inv = ((1.0 - img_float) * 255).round().astype(np.uint8)
    return img_inv


class Controller:
    """Ha's controller: W·[z,c,h] + b -> continuous action in [-1,1].

    1088 params: W is (1, 1088), no bias (or bias folded in).
    Output is tanh-clipped to [-1, 1].
    """

    def __init__(self, weights):
        self.weights = np.asarray(weights, dtype=np.float32).flatten()

    def act(self, z, h, c):
        """Returns continuous action in [-1, 1]."""
        obs = np.concatenate([z, c, h])  # Ha: [z, c, h]
        out = float(np.dot(self.weights, obs))
        return np.tanh(out)  # Ha clips to [-1, 1] via action space


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../outputs/vizdoom/ha_exact")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--render", action="store_true", help="Show VizDoom window")
    parser.add_argument("--zonly", action="store_true", help="Use z-only controller")
    args = parser.parse_args()

    device = torch.device("cpu")

    # Load models
    vae = ConvVAE(latent_dim=64, kl_tolerance=0.5).to(device)
    vae.load_state_dict(
        torch.load(os.path.join(args.data_dir, "vae.pth"), map_location=device)
    )
    vae.eval()

    rnn = MDNRNN(z_size=64, n_mix=5, rnn_size=512).to(device)
    rnn.load_state_dict(
        torch.load(os.path.join(args.data_dir, "mdn_rnn.pth"), map_location=device)
    )
    rnn.eval()

    # Load controller
    if args.zonly:
        from train_controller_ha import ZOnlyController

        ctrl_path = os.path.join(args.data_dir, "controller_zonly_best.npy")
        if not os.path.exists(ctrl_path):
            ctrl_path = os.path.join(args.data_dir, "controller_zonly_params.npy")
        weights = np.load(ctrl_path)
        controller = ZOnlyController(weights)
        use_rnn_state = False
        print(f"Z-only controller loaded from {ctrl_path}")
    else:
        ctrl_path = os.path.join(args.data_dir, "controller_best.npy")
        if not os.path.exists(ctrl_path):
            ctrl_path = os.path.join(args.data_dir, "controller_params.npy")
        weights = np.load(ctrl_path)
        controller = Controller(weights)
        use_rnn_state = True
        print(f"Full controller loaded from {ctrl_path}")

    # Set up VizDoom
    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, "take_cover.cfg"))
    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_window_visible(args.render)
    game.init()

    print(f"\nRunning {args.episodes} episodes...")
    all_steps = []

    for ep in range(args.episodes):
        game.new_episode()
        state = rnn.init_state(1, device)
        restart = 1.0
        steps = 0

        while not game.is_episode_finished():
            screen = game.get_state().screen_buffer

            # Process frame exactly like Ha
            frame = process_frame(screen)
            frame_t = (
                torch.from_numpy(frame).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            )

            # Encode with VAE (Ha uses stochastic z)
            with torch.no_grad():
                mu, logvar = vae.encode(frame_t)
                z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

            z_np = z[0].numpy()

            # Controller action
            if use_rnn_state:
                h_np = state[0][0].numpy()
                c_np = state[1][0].numpy()
                action = controller.act(z_np, h_np, c_np)  # continuous [-1, 1]
            else:
                action = controller.act(z_np)  # z-only returns continuous

            # Feed CONTINUOUS action to RNN (Ha's key design)
            with torch.no_grad():
                _, _, _, rl, state = rnn.forward_step(
                    z, action, restart, state  # action is continuous float
                )
            restart = 1.0 if rl > 0 else 0.0

            # Threshold action for VizDoom (Ha doomreal.py lines 90-97)
            threshold = 0.3333
            if action < -threshold:
                vzd_action = [1, 0]  # MOVE_LEFT
            elif action > threshold:
                vzd_action = [0, 1]  # MOVE_RIGHT
            else:
                vzd_action = [0, 0]  # stay

            game.make_action(vzd_action)
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
    print(f"\n  Ha's paper (full C):   1092 ± 556")
    print(f"  Ha's z-only+hidden:    788 ± 141")
    print(f"  Random policy:         ~210")
    print(f"  Solve threshold:       750")
