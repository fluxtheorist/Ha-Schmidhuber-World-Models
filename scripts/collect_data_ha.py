#!/usr/bin/env python3
"""
Exact port of Ha's extract.py data collection.

Key differences from our previous collection:
1. Continuous actions in [-1, 1] (not discrete {0,1,2})
2. Image inversion: (1.0 - img) * 255
3. Action persistence: same action held for 1-10 frames
4. Frame cropping: top 400 rows of 480 before resize (removes HUD)
5. Action threshold: <-0.333 = left, >0.333 = right, else stay

Usage:
    python collect_data_ha.py --episodes 10000
"""

import os
import argparse
import time
import numpy as np
import vizdoom as vzd
from PIL import Image

SCREEN_Y = 64
SCREEN_X = 64
MAX_FRAMES = 2100
MIN_LENGTH = 100


def process_frame(screen_buffer):
    """Ha's _process_frame from doomreal.py lines 21-25.

    1. Crop to top 400 rows (removes HUD)
    2. Resize to 64x64
    3. Invert colors: (1.0 - img) * 255
    """
    # screen_buffer is (120, 160, 3) from RES_160X120
    # Ha's original was (480, 640, 3) cropped to (400, 640, 3)
    # For 120 height: crop to top 100 rows (same ratio: 400/480 ≈ 0.833)
    crop_height = int(screen_buffer.shape[0] * 400 / 480)
    cropped = screen_buffer[:crop_height, :, :]

    # Resize to 64x64
    img = np.array(Image.fromarray(cropped).resize((SCREEN_X, SCREEN_Y)))

    # Invert: Ha's (1.0 - obs) * 255
    img_float = img.astype(np.float32) / 255.0
    img_inv = ((1.0 - img_float) * 255).round().astype(np.uint8)

    return img_inv


def action_to_vizdoom(action_float):
    """Ha's doomreal.py lines 90-97: continuous action -> VizDoom buttons.

    action < -0.333 -> MOVE_LEFT (button 0)
    action > 0.333  -> MOVE_RIGHT (button 1)
    else            -> stay (no buttons)
    """
    threshold = 0.3333
    if action_float < -threshold:
        return [1, 0]  # MOVE_LEFT
    elif action_float > threshold:
        return [0, 1]  # MOVE_RIGHT
    else:
        return [0, 0]  # stay


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--output_dir", type=str, default="../outputs/vizdoom/ha_exact")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, "take_cover.cfg"))
    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_window_visible(False)
    game.init()

    all_frames = []
    all_actions = []
    all_dones = []
    episode_lengths = []
    total_frames = 0
    start_time = time.time()

    for episode in range(args.episodes):
        game.new_episode()

        ep_frames = []
        ep_actions = []

        # Ha's action persistence: hold action for 1-10 frames
        repeat = np.random.randint(1, 11)
        action = 0.0  # initial action

        for frame_idx in range(MAX_FRAMES):
            if game.is_episode_finished():
                break

            state = game.get_state()
            screen = state.screen_buffer

            # Process frame exactly like Ha
            processed = process_frame(screen)
            ep_frames.append(processed)

            # Random continuous action with persistence (Ha extract.py lines 54-56)
            if frame_idx % repeat == 0:
                action = np.random.rand() * 2.0 - 1.0
                repeat = np.random.randint(1, 11)

            ep_actions.append(action)

            # Execute in VizDoom
            vzd_action = action_to_vizdoom(action)
            game.make_action(vzd_action)

        ep_length = len(ep_frames)

        if ep_length > MIN_LENGTH:
            # Build done flags
            ep_dones = [0.0] * ep_length
            if game.is_player_dead():
                ep_dones[-1] = 1.0

            all_frames.extend(ep_frames)
            all_actions.extend(ep_actions)
            all_dones.extend(ep_dones)
            episode_lengths.append(ep_length)
            total_frames += ep_length

        # Save chunks every 1000 episodes
        if (episode + 1) % 1000 == 0 and all_frames:
            chunk_num = (episode + 1) // 1000
            np.save(
                f"{args.output_dir}/frames_chunk{chunk_num}.npy",
                np.array(all_frames, dtype=np.uint8),
            )
            np.save(
                f"{args.output_dir}/actions_chunk{chunk_num}.npy",
                np.array(all_actions, dtype=np.float32),
            )  # CONTINUOUS float, not int!
            np.save(
                f"{args.output_dir}/dones_chunk{chunk_num}.npy",
                np.array(all_dones, dtype=np.float32),
            )

            elapsed = time.time() - start_time
            avg_len = np.mean(episode_lengths[-1000:])
            print(
                f"Episode {episode+1}/{args.episodes} | "
                f"Avg length: {avg_len:.0f} | "
                f"Total frames: {total_frames:,} | "
                f"Chunk {chunk_num} saved | {elapsed:.0f}s"
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
            np.array(all_actions, dtype=np.float32),
        )
        np.save(
            f"{args.output_dir}/dones_chunk{chunk_num}.npy",
            np.array(all_dones, dtype=np.float32),
        )

    np.save(f"{args.output_dir}/episode_lengths.npy", np.array(episode_lengths))

    print(f"\nCollection complete!")
    print(f"Episodes: {len(episode_lengths)}")
    print(f"Total frames: {total_frames:,}")
    print(f"Avg length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Actions: continuous float32 in [-1, 1]")
    print(f"Frames: inverted, cropped, 64x64")
