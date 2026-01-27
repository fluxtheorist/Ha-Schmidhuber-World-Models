import sys

sys.path.append("..")

import torch
import numpy as np
import gymnasium as gym
import argparse
from PIL import Image
from multiprocessing import Pool, cpu_count
import cma

from models.vae import ConvVAE
from models.mdn_rnn import MDNRNN
from models.controller import Controller


def set_controller_params(controller, params, device):
    """Load numpy params into controller"""
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


def rollout(params, vae, mdn_rnn, controller, device, max_steps=500, render=False):
    """Run one episode and return total reward"""
    set_controller_params(controller, params, device)

    env = gym.make(
        "CarRacing-v3", continuous=True, render_mode="human" if render else None
    )
    obs, _ = env.reset()
    obs = np.array(Image.fromarray(obs).resize((64, 64)))

    hidden = (torch.zeros(1, 1, 256).to(device), torch.zeros(1, 1, 256).to(device))
    total_reward = 0

    for step in range(max_steps):
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).float() / 255.0
            obs_tensor = obs_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
            mu, _ = vae.encode(obs_tensor)
            z = mu.squeeze(0)

            h = hidden[0].squeeze(0).squeeze(0)
            action = controller(z, h).cpu().numpy()

            action_env = np.array([action[0], (action[1] + 1) / 2, (action[2] + 1) / 2])

            obs, reward, terminated, truncated, _ = env.step(action_env)
            obs = np.array(Image.fromarray(obs).resize((64, 64)))
            total_reward += reward

            z_input = z.unsqueeze(0).unsqueeze(0)
            a_tensor = torch.from_numpy(action).float().to(device)
            a_input = a_tensor.unsqueeze(0).unsqueeze(0)
            _, hidden = mdn_rnn(z_input, a_input, hidden)

        if terminated or truncated:
            break

    env.close()
    return total_reward


# Global models for worker processes (loaded once per worker)
_worker_models = None


def init_worker(iter_num):
    """Initialize models in each worker process (called once per worker)"""
    global _worker_models

    device = torch.device("cpu")  # Workers use CPU for parallel execution
    load_dir = f"../outputs/iter{iter_num}"

    vae = ConvVAE(latent_dim=32)
    vae.load_state_dict(torch.load(f"{load_dir}/vae.pth", map_location=device))
    vae.to(device)
    vae.eval()

    mdn_rnn = MDNRNN(latent_dim=32, action_dim=3, hidden_dim=256, n_gaussians=5)
    mdn_rnn.load_state_dict(torch.load(f"{load_dir}/mdn_rnn.pth", map_location=device))
    mdn_rnn.to(device)
    mdn_rnn.eval()

    controller = Controller()
    controller.to(device)
    controller.eval()

    _worker_models = (vae, mdn_rnn, controller, device)


def evaluate_worker(args):
    """Evaluate a single solution (runs in worker process)"""
    params, max_steps, n_rollouts = args

    vae, mdn_rnn, controller, device = _worker_models

    # Average over multiple rollouts for stability
    rewards = []
    for _ in range(n_rollouts):
        reward = rollout(params, vae, mdn_rnn, controller, device, max_steps)
        rewards.append(reward)

    return np.mean(rewards)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=0, help="Iteration number")
    parser.add_argument("--gens", type=int, default=50, help="Number of generations")
    parser.add_argument(
        "--max_steps", type=int, default=500, help="Max steps per rollout"
    )
    parser.add_argument(
        "--rollouts", type=int, default=4, help="Rollouts per evaluation (averaged)"
    )
    parser.add_argument(
        "--workers", type=int, default=10, help="Number of parallel workers"
    )
    parser.add_argument(
        "--popsize", type=int, default=16, help="Population size for CMA-ES"
    )
    args = parser.parse_args()

    output_dir = f"../outputs/iter{args.iter}"

    # Count controller parameters
    temp_controller = Controller()
    n_params = sum(p.numel() for p in temp_controller.parameters())
    print(f"Controller has {n_params} parameters")
    print(f"Using {args.workers} parallel workers")
    print(f"Population size: {args.popsize}, Generations: {args.gens}")
    print(f"Max steps: {args.max_steps}, Rollouts per eval: {args.rollouts}")

    # Initialize CMA-ES
    es = cma.CMAEvolutionStrategy(
        n_params * [0],  # Start from zeros
        0.5,  # Initial sigma
        {"popsize": args.popsize},
    )

    # Track history for plotting
    history = {"generation": [], "best": [], "mean": [], "worst": []}

    # Create worker pool with initialized models
    print(f"\nInitializing {args.workers} workers...")
    pool = Pool(processes=args.workers, initializer=init_worker, initargs=(args.iter,))

    try:
        gen = 0
        while not es.stop() and gen < args.gens:
            # Get candidate solutions
            solutions = es.ask()

            # Evaluate in parallel
            args_list = [(sol, args.max_steps, args.rollouts) for sol in solutions]
            rewards = pool.map(evaluate_worker, args_list)

            # CMA-ES minimizes, so negate rewards
            fitness = [-r for r in rewards]
            es.tell(solutions, fitness)

            # Track stats
            best_reward = max(rewards)
            mean_reward = np.mean(rewards)
            worst_reward = min(rewards)

            history["generation"].append(gen)
            history["best"].append(best_reward)
            history["mean"].append(mean_reward)
            history["worst"].append(worst_reward)

            print(
                f"Gen {gen:3d} | Best: {best_reward:7.1f} | Mean: {mean_reward:7.1f} | Worst: {worst_reward:7.1f}"
            )

            # Save best so far every 10 generations
            if gen % 10 == 0:
                np.save(f"{output_dir}/controller_params.npy", es.result.xbest)

            gen += 1

    finally:
        pool.close()
        pool.join()

    # Save final results
    np.save(f"{output_dir}/controller_params.npy", es.result.xbest)
    np.save(f"{output_dir}/training_history.npy", history)

    print(f"\nTraining complete!")
    print(f"Saved controller to {output_dir}/controller_params.npy")
    print(f"Saved history to {output_dir}/training_history.npy")

    # Quick summary
    print(f"\nFinal best reward: {max(history['best']):.1f}")
    print(f"Final mean reward: {history['mean'][-1]:.1f}")
