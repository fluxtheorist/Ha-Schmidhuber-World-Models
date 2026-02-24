#!/usr/bin/env python3
"""Quick diagnostic: test controller with different settings to isolate the problem."""

import sys

sys.path.append("..")
import numpy as np
import torch
import vizdoom as vzd
from PIL import Image
from models.vae import ConvVAE
from models.mdn_rnn import MDNRNN

device = torch.device("cpu")

vae = ConvVAE(latent_dim=64, kl_tolerance=0.5).to(device)
vae.load_state_dict(torch.load("../outputs/vizdoom/iter0/vae.pth", map_location=device))
vae.eval()

rnn = MDNRNN(z_size=64, n_mix=5, rnn_size=512).to(device)
rnn.load_state_dict(
    torch.load("../outputs/vizdoom/iter0/mdn_rnn.pth", map_location=device)
)
rnn.eval()

w = np.load("../outputs/vizdoom/iter0/controller_params.npy")

game = vzd.DoomGame()
game.load_config(vzd.scenarios_path + "/take_cover.cfg")
game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
game.set_screen_format(vzd.ScreenFormat.RGB24)
game.set_window_visible(False)
game.init()

actions_list = [[0, 0], [1, 0], [0, 1]]


def run_test(mode, episodes=20):
    results = []
    for ep in range(episodes):
        game.new_episode()
        state = rnn.init_state(1, device)
        restart = 1.0
        steps = 0
        action_counts = [0, 0, 0]

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

            if mode == "normal":
                h_np = state[0][0].numpy()
                c_np = state[1][0].numpy()
            elif mode == "zero_state":
                h_np = np.zeros(512)
                c_np = np.zeros(512)
            elif mode == "z_only":
                # Only use z weights, zero out h and c contribution
                h_np = np.zeros(512)
                c_np = np.zeros(512)
            elif mode == "random":
                action_idx = np.random.randint(3)
                game.make_action(actions_list[action_idx], 4)
                steps += 1
                continue

            obs = np.concatenate([z_np, c_np, h_np])
            out = float(np.dot(w, obs))
            if out < -0.33:
                action_idx = 0
            elif out > 0.33:
                action_idx = 2
            else:
                action_idx = 1
            action_counts[action_idx] += 1

            game.make_action(actions_list[action_idx])
            steps += 1

            if mode == "normal":
                with torch.no_grad():
                    _, _, _, rl, state = rnn.forward_step(
                        z, float(action_idx), restart, state
                    )
                restart = 1.0 if rl > 0 else 0.0

        results.append(steps)

    avg = np.mean(results)
    std = np.std(results)
    print(
        f"  {mode:>12}: {avg:.1f} ± {std:.1f}  (actions: L={action_counts[0]}, S={action_counts[1]}, R={action_counts[2]} last ep)"
    )
    return avg


print("Testing different controller modes (20 episodes each):\n")
run_test("normal")
run_test("zero_state")
run_test("z_only")

game.close()
