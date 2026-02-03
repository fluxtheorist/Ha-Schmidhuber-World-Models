"""
Test VizDoom Take Cover environment setup.
Verifies: scenario loading, observation format, action space, episode mechanics.

Run this on your Mac to make sure everything works before we start collecting data.
"""

import vizdoom as vzd
import numpy as np
from PIL import Image
import os
import time


def test_native_api():
    """Test using VizDoom's native Python API (what we'll use for data collection)."""
    print("=" * 60)
    print("TEST 1: VizDoom Native API - Take Cover")
    print("=" * 60)

    game = vzd.DoomGame()

    # Load Take Cover scenario
    scenario_path = os.path.join(vzd.scenarios_path, "take_cover.cfg")
    print(f"Scenario path: {scenario_path}")
    print(f"Exists: {os.path.exists(scenario_path)}")
    game.load_config(scenario_path)

    # Set screen format and resolution
    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    game.set_screen_format(vzd.ScreenFormat.RGB24)

    # Don't render window (headless for data collection)
    game.set_window_visible(False)

    # Print available buttons before init
    print(f"\nAvailable buttons: {game.get_available_buttons()}")

    game.init()

    # Check what we're working with
    print(f"\n--- Environment Info ---")
    print(f"Available buttons: {game.get_available_buttons()}")
    print(f"Number of available buttons: {game.get_available_buttons_size()}")
    print(f"Available game variables: {game.get_available_game_variables()}")
    print(f"Episode timeout: {game.get_episode_timeout()}")

    # Define our 3 discrete actions: [MOVE_LEFT, MOVE_RIGHT]
    # Action 0: Do nothing  [0, 0]
    # Action 1: Move left   [1, 0]
    # Action 2: Move right  [0, 1]
    actions = [[0, 0], [1, 0], [0, 1]]
    action_names = ["STAY", "MOVE_LEFT", "MOVE_RIGHT"]

    # Run one episode
    game.new_episode()
    total_reward = 0
    step = 0

    print(f"\n--- Running Episode ---")

    while not game.is_episode_finished():
        state = game.get_state()

        # Check observation format
        if step == 0:
            screen = state.screen_buffer  # numpy array
            print(f"\nRaw screen shape: {screen.shape}")
            print(f"Screen dtype: {screen.dtype}")
            print(f"Screen range: [{screen.min()}, {screen.max()}]")

            # VizDoom returns (C, H, W) format — need to transpose for PIL
            if screen.ndim == 3 and screen.shape[0] == 3:
                screen_hwc = np.transpose(screen, (1, 2, 0))
            else:
                screen_hwc = screen

            print(f"Screen (H,W,C): {screen_hwc.shape}")

            # Resize to 64x64 like we'll do for VAE
            img = Image.fromarray(screen_hwc)
            img_64 = img.resize((64, 64))
            img_arr = np.array(img_64)
            print(f"Resized to 64x64: {img_arr.shape}")

            # Check game variables
            print(f"\nGame variables: {state.game_variables}")
            print(f"  (Usually [HEALTH] for Take Cover)")

        # Take random action
        action_idx = np.random.randint(len(actions))
        reward = game.make_action(actions[action_idx])
        total_reward += reward
        step += 1

        # Print status every 100 steps
        if step % 100 == 0:
            if not game.is_episode_finished():
                health = (
                    state.game_variables[0]
                    if state.game_variables is not None
                    else "N/A"
                )
                print(
                    f"  Step {step}: reward={reward:.1f}, total={total_reward:.1f}, health={health}"
                )

    print(f"\n--- Episode Complete ---")
    print(f"Total steps survived: {step}")
    print(f"Total reward: {total_reward:.1f}")
    print(f"Episode finished: {game.is_episode_finished()}")
    print(f"Player dead: {game.is_player_dead()}")

    game.close()
    return step


def test_gymnasium_wrapper():
    """Test using Gymnasium wrapper (alternative interface)."""
    print("\n" + "=" * 60)
    print("TEST 2: Gymnasium Wrapper - VizdoomTakeCover-v0")
    print("=" * 60)

    try:
        import gymnasium
        from vizdoom import gymnasium_wrapper

        env = gymnasium.make("VizdoomTakeCover-v0", render_mode=None)

        obs, info = env.reset()
        print(f"\nObservation type: {type(obs)}")
        if isinstance(obs, dict):
            for key, val in obs.items():
                print(f"  '{key}': shape={val.shape}, dtype={val.dtype}")
        else:
            print(f"  shape={obs.shape}, dtype={obs.dtype}")

        print(f"Action space: {env.action_space}")
        print(f"Observation space: {env.observation_space}")

        # Run quick episode
        total_reward = 0
        steps = 0
        done = False

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        print(f"\nSteps survived: {steps}")
        print(f"Total reward: {total_reward:.1f}")
        print(f"Terminated (death): {terminated}")
        print(f"Truncated (timeout): {truncated}")

        env.close()

    except Exception as e:
        print(f"Gymnasium wrapper test failed: {e}")
        print("(This is fine - we'll use the native API anyway)")


def test_speed():
    """Benchmark: how fast can we collect frames?"""
    print("\n" + "=" * 60)
    print("TEST 3: Collection Speed Benchmark")
    print("=" * 60)

    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, "take_cover.cfg"))
    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_window_visible(False)
    game.init()

    actions = [[0, 0], [1, 0], [0, 1]]

    total_frames = 0
    start = time.time()
    n_episodes = 10

    for ep in range(n_episodes):
        game.new_episode()
        while not game.is_episode_finished():
            state = game.get_state()
            screen = state.screen_buffer

            # Simulate what we'll do: resize to 64x64
            if screen.ndim == 3 and screen.shape[0] == 3:
                screen_hwc = np.transpose(screen, (1, 2, 0))
            else:
                screen_hwc = screen

            img = Image.fromarray(screen_hwc).resize((64, 64))
            frame = np.array(img)

            action_idx = np.random.randint(len(actions))
            game.make_action(actions[action_idx])
            total_frames += 1

    elapsed = time.time() - start
    fps = total_frames / elapsed

    print(f"Collected {total_frames} frames in {elapsed:.2f}s")
    print(f"Speed: {fps:.0f} FPS")
    print(f"Episodes: {n_episodes}, Avg frames/episode: {total_frames/n_episodes:.0f}")

    # Estimate time for 10,000 episodes (paper's dataset size)
    est_frames = (total_frames / n_episodes) * 10000
    est_time = est_frames / fps
    print(
        f"\nEstimate for 10,000 episodes: ~{est_frames:.0f} frames, ~{est_time/60:.1f} minutes"
    )

    game.close()


def list_scenarios():
    """List all available VizDoom scenarios."""
    print("\n" + "=" * 60)
    print("Available VizDoom Scenarios")
    print("=" * 60)
    print(f"Scenarios path: {vzd.scenarios_path}")

    for f in sorted(os.listdir(vzd.scenarios_path)):
        print(f"  {f}")


if __name__ == "__main__":
    list_scenarios()
    test_native_api()
    test_gymnasium_wrapper()
    test_speed()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)
    print("\nKey takeaways for World Models implementation:")
    print("  - Screen format: RGB24, needs resize to 64x64")
    print("  - VizDoom returns (C, H, W) — transpose to (H, W, C) before PIL")
    print("  - Actions: 3 discrete [STAY, LEFT, RIGHT]")
    print("  - Reward = +1 per timestep alive (survival time)")
    print("  - Episode ends on death or timeout (~2100 steps)")
    print("  - VAE latent dim: 64 (not 32 like CarRacing)")
    print("  - LSTM hidden: 512 (not 256)")
    print("  - Controller uses h AND c from LSTM")
