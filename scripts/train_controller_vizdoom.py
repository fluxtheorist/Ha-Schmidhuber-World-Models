#!/usr/bin/env python3
"""
CMA-ES controller training in MDN-RNN dream environment.
Matches Ha's DoomCoverRNNEnv and controller architecture.

Controller: linear map from [z(64), h(512), c(512)] = 1088 -> scalar action
Dream env: MDN-RNN generates next z, restart_logit > 0 means death
Reward: +1 per step survived
Temperature: 1.25

Usage:
    python train_controller_vizdoom.py --data_dir ../outputs/vizdoom/iter0
"""

import sys

sys.path.append("..")

import os
import argparse
import time
import multiprocessing
import numpy as np
import torch

from models.mdn_rnn import MDNRNN

# ================================================================
# Controller: 1088 weights -> action (Ha's architecture)
# ================================================================


class Controller:
    """Linear controller: [z, h, c] -> action.

    Ha's observation (line 576): concatenate(z, c, h)  [note: c before h]
    Action: scalar in [-1, 1], discretized to {0=left, 1=stay, 2=right}
    """

    def __init__(self, weights=None):
        self.input_dim = 64 + 512 + 512  # z + c + h = 1088
        if weights is None:
            self.weights = np.zeros(self.input_dim, dtype=np.float32)
        else:
            self.weights = np.asarray(weights, dtype=np.float32).flatten()

    def act(self, z, h, c):
        """Compute action from current state.

        Args:
            z: (64,) numpy array
            h: (512,) numpy array
            c: (512,) numpy array

        Returns:
            action: int (0, 1, or 2)
            action_float: float for feeding to RNN
        """
        # Ha line 576: concatenate(z, c, h) — note c before h
        obs = np.concatenate([z, c, h])
        out = float(np.dot(self.weights, obs))

        # Discretize: Ha maps continuous [-1,1] to 3 actions
        if out < -0.33:
            return 0, 0.0  # left
        elif out > 0.33:
            return 2, 2.0  # right
        else:
            return 1, 1.0  # stay


# ================================================================
# Dream rollout matching Ha's DoomCoverRNNEnv._step
# ================================================================

TEMPERATURE = 1.15  # Ha appendix A.5: tau=1.15 gave best real transfer (1092±556)
MAX_FRAMES = 2100  # Ha line 560
ENTROPY_BONUS = (
    200  # Bonus scaling for action diversity (max entropy ≈ 1.1, so max bonus ≈ 220)
)


def dream_rollout(rnn, controller, initial_z_data, device):
    """Run one episode in the dream environment.

    Returns (steps_survived, action_counts) where action_counts is [left, stay, right].
    """
    rnn.eval()
    action_counts = [0, 0, 0]

    with torch.no_grad():
        # Sample initial z (Ha lines 566-572)
        idx = np.random.randint(0, len(initial_z_data["mu"]))
        init_mu = initial_z_data["mu"][idx]
        init_logvar = initial_z_data["logvar"][idx]
        z = init_mu + np.exp(init_logvar / 2.0) * np.random.randn(*init_logvar.shape)

        # Initial state (Ha lines 579-583)
        state = rnn.init_state(1, device)
        restart = 1.0  # First step is a restart

        for step in range(MAX_FRAMES):
            # Get h, c as numpy for controller
            h_np = state[0][0].cpu().numpy()  # (rnn_size,)
            c_np = state[1][0].cpu().numpy()  # (rnn_size,)

            # Controller action
            action_idx, action_float = controller.act(z, h_np, c_np)
            action_counts[action_idx] += 1

            # RNN step
            z_tensor = torch.from_numpy(z).float().unsqueeze(0).to(device)  # (1, 64)
            logmix, mean, logstd, restart_logit, state = rnn.forward_step(
                z_tensor, action_float, restart, state
            )

            # Sample next z
            z = rnn.sample_z(logmix, mean, logstd, temperature=TEMPERATURE)

            # Check termination (Ha line 644: logrestart[0] > 0)
            if restart_logit > 0:
                restart = 1.0
                return step + 1, action_counts
            else:
                restart = 0.0

        return MAX_FRAMES, action_counts


# ================================================================
# Worker pool for parallel evaluation
# ================================================================

_WORKER_RNN = None
_WORKER_INIT_Z = None
_WORKER_DEVICE = None


def init_worker(rnn_path, init_z_path, device_str):
    global _WORKER_RNN, _WORKER_INIT_Z, _WORKER_DEVICE

    _WORKER_DEVICE = torch.device(device_str)

    _WORKER_RNN = MDNRNN(z_size=64, n_mix=5, rnn_size=512).to(_WORKER_DEVICE)
    ckpt = torch.load(rnn_path, map_location=_WORKER_DEVICE)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    _WORKER_RNN.load_state_dict(ckpt)
    _WORKER_RNN.eval()

    data = np.load(init_z_path)
    _WORKER_INIT_Z = {"mu": data["mu"], "logvar": data["logvar"]}


def evaluate_worker(args):
    weights, num_rollouts = args
    controller = Controller(weights=weights)

    fitnesses = []
    for _ in range(num_rollouts):
        steps, action_counts = dream_rollout(
            _WORKER_RNN, controller, _WORKER_INIT_Z, _WORKER_DEVICE
        )

        # Entropy bonus: reward action diversity to prevent degenerate strategies
        total_actions = sum(action_counts)
        if total_actions > 0:
            probs = [c / total_actions for c in action_counts]
            entropy = -sum(p * np.log(p + 1e-10) for p in probs)
        else:
            entropy = 0.0
        # Max entropy for 3 actions = log(3) ≈ 1.099
        # Scale so entropy bonus is significant but doesn't dominate survival
        fitness = steps + ENTROPY_BONUS * entropy
        fitnesses.append(fitness)

    return float(np.mean(fitnesses))


# ================================================================
# Main: CMA-ES
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../outputs/vizdoom/iter0")
    parser.add_argument("--output", type=str, default="../outputs/vizdoom/iter0")
    parser.add_argument("--gens", type=int, default=300)
    parser.add_argument("--popsize", type=int, default=64)
    parser.add_argument(
        "--rollouts", type=int, default=16, help="Rollouts per candidate per generation"
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--sigma", type=float, default=0.5, help="Initial CMA-ES step size"
    )
    args = parser.parse_args()

    import cma

    os.makedirs(args.output, exist_ok=True)

    rnn_path = os.path.join(args.data_dir, "mdn_rnn.pth")
    init_z_path = os.path.join(args.data_dir, "initial_z.npz")

    assert os.path.exists(rnn_path), f"MDN-RNN not found: {rnn_path}"
    assert os.path.exists(init_z_path), f"initial_z not found: {init_z_path}"

    n_params = 1088  # z(64) + c(512) + h(512)
    print(f"Controller params: {n_params}")
    print(f"CMA-ES: popsize={args.popsize}, σ={args.sigma}, gens={args.gens}")
    print(f"Rollouts per eval: {args.rollouts}")
    print(f"Workers: {args.workers}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Max frames: {MAX_FRAMES}")

    # Quick sanity: run one rollout
    print("\nSanity check: running one dream rollout...")
    rnn = MDNRNN(z_size=64, n_mix=5, rnn_size=512)
    rnn.load_state_dict(torch.load(rnn_path, map_location="cpu"))
    rnn.eval()
    init_z = np.load(init_z_path)
    init_z_dict = {"mu": init_z["mu"], "logvar": init_z["logvar"]}

    test_ctrl = Controller()  # zero weights
    test_reward, test_actions = dream_rollout(
        rnn, test_ctrl, init_z_dict, torch.device("cpu")
    )
    print(f"  Zero-weight controller survived {test_reward} steps")
    print(
        f"  Actions: left={test_actions[0]}, stay={test_actions[1]}, right={test_actions[2]}"
    )
    del rnn

    # Start CMA-ES
    es = cma.CMAEvolutionStrategy(
        n_params * [0.0],
        args.sigma,
        {"popsize": args.popsize},
    )

    print(f"\nInitializing {args.workers} worker processes...")
    pool = multiprocessing.Pool(
        processes=args.workers,
        initializer=init_worker,
        initargs=(rnn_path, init_z_path, "cpu"),
    )

    history = {"gen": [], "best": [], "mean": [], "worst": []}
    best_ever = 0

    try:
        for gen in range(args.gens):
            t0 = time.time()

            solutions = es.ask()
            eval_args = [(w, args.rollouts) for w in solutions]
            rewards = pool.map(evaluate_worker, eval_args)

            # CMA-ES minimizes, negate rewards
            es.tell(solutions, [-r for r in rewards])

            best = max(rewards)
            mean = np.mean(rewards)
            worst = min(rewards)

            if best > best_ever:
                best_ever = best
                np.save(
                    os.path.join(args.output, "controller_best.npy"), es.result.xbest
                )

            history["gen"].append(gen)
            history["best"].append(best)
            history["mean"].append(mean)
            history["worst"].append(worst)

            elapsed = time.time() - t0
            print(
                f"Gen {gen:3d} | Best: {best:7.1f} | Mean: {mean:7.1f} | "
                f"Worst: {worst:7.1f} | Best-ever: {best_ever:7.1f} | {elapsed:.1f}s"
            )

            # Checkpoint every 25 gens
            if gen % 25 == 0:
                np.save(
                    os.path.join(args.output, "controller_params.npy"), es.result.xbest
                )
                np.savez(os.path.join(args.output, "controller_history.npz"), **history)

    finally:
        pool.close()
        pool.join()

    # Final save
    np.save(os.path.join(args.output, "controller_params.npy"), es.result.xbest)
    np.savez(os.path.join(args.output, "controller_history.npz"), **history)

    print(f"\nTraining complete!")
    print(f"Best-ever survival: {best_ever:.1f} steps")
    print(f"Final mean: {history['mean'][-1]:.1f} steps")
