#!/usr/bin/env python3
"""
Render a dream rollout by decoding z vectors through the VAE.
Shows what the agent 'imagines' during dream training.

Usage:
    python render_dream.py --data_dir ../outputs/vizdoom/ha_exact --steps 500
"""

import sys

sys.path.append("..")

import os
import argparse
import numpy as np
import torch
from PIL import Image

from models.vae import ConvVAE
from models.mdn_rnn import MDNRNN
from train_controller_ha import FullController

TEMPERATURE = 1.15


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../outputs/vizdoom/ha_exact")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument(
        "--output", type=str, default="../outputs/vizdoom/ha_exact/dream_frames"
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
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

    weights = np.load(os.path.join(args.data_dir, "controller_best.npy"))
    controller = FullController(weights=weights)

    init_z_data = np.load(os.path.join(args.data_dir, "initial_z.npz"))

    # Sample initial z
    idx = np.random.randint(0, len(init_z_data["mu"]))
    init_mu = init_z_data["mu"][idx]
    init_logvar = init_z_data["logvar"][idx]
    z = init_mu + np.exp(init_logvar / 2.0) * np.random.randn(*init_logvar.shape)

    state = rnn.init_state(1, device)
    restart = 1.0

    frames = []
    print(f"Running dream rollout for {args.steps} steps...")

    with torch.no_grad():
        for step in range(args.steps):
            # Decode z to image
            z_tensor = torch.from_numpy(z).float().unsqueeze(0).to(device)
            decoded = vae.decode(z_tensor)  # (1, 3, 64, 64)
            img = (
                (decoded[0].permute(1, 2, 0).numpy() * 255)
                .clip(0, 255)
                .astype(np.uint8)
            )

            # Un-invert the colors (Ha inverted during collection)
            img = 255 - img

            # Upscale for visibility
            img_large = np.array(Image.fromarray(img).resize((256, 256), Image.NEAREST))
            frames.append(img_large)

            # Controller action
            h_np = state[0][0].numpy()
            c_np = state[1][0].numpy()
            action = controller.act(z, h_np, c_np)

            # RNN step
            logmix, mean, logstd, restart_logit, state = rnn.forward_step(
                z_tensor, action, restart, state
            )

            z = rnn.sample_z(logmix, mean, logstd, temperature=TEMPERATURE)

            if restart_logit > 0:
                print(f"  Agent died at step {step + 1}")
                break
            else:
                restart = 0.0

            if step % 50 == 0:
                # Save action info
                action_str = (
                    "LEFT"
                    if action < -0.333
                    else ("RIGHT" if action > 0.333 else "STAY")
                )
                print(f"  Step {step}: action={action:.3f} ({action_str})")

    print(f"Survived {len(frames)} steps in dream")

    # Save as individual frames
    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(os.path.join(args.output, f"frame_{i:04d}.png"))

    # Save as GIF
    gif_path = os.path.join(args.data_dir, "dream_rollout.gif")
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=33,  # ~30 fps
        loop=0,
    )
    print(f"Saved GIF to {gif_path}")
    print(f"Saved {len(frames)} frames to {args.output}/")
