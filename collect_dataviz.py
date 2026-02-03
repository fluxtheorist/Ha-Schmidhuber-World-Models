import vizdoom as vzd
import numpy as np
from PIL import Image
import os
import argparse
import time

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=10000)
parser.add_argument("--output_dir", type=str, default="outputs/vizdoom/iter0")
args = parser.parse_args()

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Set up game
game = vzd.DoomGame()
game.load_config(os.path.join(vzd.scenarios_path, "take_cover.cfg"))
game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
game.set_screen_format(vzd.ScreenFormat.RGB24)
game.set_window_visible(False)
game.init()

# Action space
actions = [[0, 0], [1, 0], [0, 1]]

# Storage arrays
all_frames = []
all_actions = []
all_dones = []
episode_lengths = []

# Start time
start_time = time.time()

# Episode loop
for episode in range(args.episodes):
    game.new_episode()
    episode_length = 0

    while not game.is_episode_finished():
        # Get game state
        state = game.get_state()
        screen = state.screen_buffer
        frame = np.array(Image.fromarray(screen).resize((64, 64)))

        # Append frame
        all_frames.append(frame)

        # Random action
        action_idx = np.random.randint(3)
        action = actions[action_idx]
        game.make_action(action)
        all_actions.append(action_idx)

        # Death label
        if game.is_episode_finished() and game.is_player_dead():
            all_dones.append(1.0)
        else:
            all_dones.append(0.0)

        episode_length += 1

    episode_lengths.append(episode_length)

    # Progress logging
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
        fps = sum(episode_lengths) / elapsed
        print(
            f"Episode {episode+1}/{args.episodes} | "
            f"Avg length: {np.mean(episode_lengths):.0f} | "
            f"Chunk {chunk_num} saved | {fps:.0f} FPS"
        )

        all_frames.clear()
        all_actions.clear()
        all_dones.clear()

# Close game
game.close()

# Save any remaining data
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

# Save episode lengths (small, keep as one file)
np.save(f"{args.output_dir}/episode_lengths.npy", np.array(episode_lengths))

print(f"\nCollection complete!")
print(f"Episodes: {len(episode_lengths)}")
print(f"Avg length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
print(f"Saved {chunk_num} chunks to {args.output_dir}/")
