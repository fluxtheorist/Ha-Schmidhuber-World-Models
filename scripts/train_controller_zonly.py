#!/usr/bin/env python3
"""
CMA-ES controller training using z-only input (no RNN hidden state).
Matches Ha's "V model with hidden layer" variant (paper score: 788).

Controller: z(64) -> hidden(40, tanh) -> action(1)
Total params: 64*40 + 40 + 40*1 + 1 = 2601 + 40 + 1 = 2641

Dream env still uses MDN-RNN for rollout, but controller only sees z.

Usage:
    python train_controller_zonly.py --data_dir ../outputs/vizdoom/iter0
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
# Controller: z -> tanh hidden -> action (no RNN state)
# ================================================================


class ZOnlyController:
    """z(64) -> Linear(40) -> tanh -> Linear(1) -> threshold -> action.

    Params layout in flat weight vector:
      W1: 64*40 = 2560
      b1: 40
      W2: 40*1 = 40
      b2: 1
      Total: 2641
    """

    def __init__(self, weights=None):
        self.z_dim = 64
        self.hidden_dim = 40
        self.n_params = (
            self.z_dim * self.hidden_dim + self.hidden_dim + self.hidden_dim + 1
        )

        if weights is None:
            self.params = np.zeros(self.n_params, dtype=np.float32)
        else:
            self.params = np.asarray(weights, dtype=np.float32).flatten()

        # Unpack weights
        idx = 0
        self.W1 = self.params[idx : idx + self.z_dim * self.hidden_dim].reshape(
            self.hidden_dim, self.z_dim
        )
        idx += self.z_dim * self.hidden_dim
        self.b1 = self.params[idx : idx + self.hidden_dim]
        idx += self.hidden_dim
        self.W2 = self.params[idx : idx + self.hidden_dim].reshape(1, self.hidden_dim)
        idx += self.hidden_dim
        self.b2 = self.params[idx : idx + 1]

    def act(self, z):
        """Compute action from z only.

        Args:
            z: (64,) numpy array

        Returns:
            action_idx: int (0, 1, or 2)
            action_float: float for feeding to RNN
        """
        hidden = np.tanh(self.W1 @ z + self.b1)
        out = float((self.W2 @ hidden + self.b2)[0])

        if out < -0.33:
            return 0, 0.0  # stay ([0,0])
        elif out > 0.33:
            return 2, 2.0  # right ([0,1])
        else:
            return 1, 1.0  # left ([1,0])


# ================================================================
# Dream rollout — RNN still runs for dynamics, but controller only sees z
# ================================================================

TEMPERATURE = 1.15
MAX_FRAMES = 2100


def dream_rollout(rnn, controller, initial_z_data, device):
    """Run one episode in dream. Controller uses z only, no RNN state."""
    rnn.eval()

    with torch.no_grad():
        idx = np.random.randint(0, len(initial_z_data["mu"]))
        init_mu = initial_z_data["mu"][idx]
        init_logvar = initial_z_data["logvar"][idx]
        z = init_mu + np.exp(init_logvar / 2.0) * np.random.randn(*init_logvar.shape)

        state = rnn.init_state(1, device)
        restart = 1.0

        for step in range(MAX_FRAMES):
            # Controller only sees z
            action_idx, action_float = controller.act(z)

            # RNN step (still needed for dream dynamics)
            z_tensor = torch.from_numpy(z).float().unsqueeze(0).to(device)
            logmix, mean, logstd, restart_logit, state = rnn.forward_step(
                z_tensor, action_float, restart, state
            )

            z = rnn.sample_z(logmix, mean, logstd, temperature=TEMPERATURE)

            if restart_logit > 0:
                return step + 1
            else:
                restart = 0.0

        return MAX_FRAMES


# ================================================================
# Worker pool
# ================================================================

_WORKER_RNN = None
_WORKER_INIT_Z = None
_WORKER_DEVICE = None


def init_worker(rnn_path, init_z_path, device_str):
    global _WORKER_RNN, _WORKER_INIT_Z, _WORKER_DEVICE

    _WORKER_DEVICE = torch.device(device_str)
    _WORKER_RNN = MDNRNN(z_size=64, n_mix=5, rnn_size=512).to(_WORKER_DEVICE)
    ckpt = torch.load(rnn_path, map_location=_WORKER_DEVICE)
    _WORKER_RNN.load_state_dict(ckpt)
    _WORKER_RNN.eval()

    data = np.load(init_z_path)
    _WORKER_INIT_Z = {"mu": data["mu"], "logvar": data["logvar"]}


def evaluate_worker(args):
    weights, num_rollouts = args
    controller = ZOnlyController(weights=weights)

    rewards = []
    for _ in range(num_rollouts):
        r = dream_rollout(_WORKER_RNN, controller, _WORKER_INIT_Z, _WORKER_DEVICE)
        rewards.append(r)

    return float(np.mean(rewards))


# ================================================================
# Main: CMA-ES
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../outputs/vizdoom/iter0")
    parser.add_argument("--output", type=str, default="../outputs/vizdoom/iter0")
    parser.add_argument("--gens", type=int, default=500)
    parser.add_argument("--popsize", type=int, default=64)
    parser.add_argument("--rollouts", type=int, default=16)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.1,
        help="Initial CMA-ES step size (smaller for hidden layer)",
    )
    args = parser.parse_args()

    import cma

    os.makedirs(args.output, exist_ok=True)

    rnn_path = os.path.join(args.data_dir, "mdn_rnn.pth")
    init_z_path = os.path.join(args.data_dir, "initial_z.npz")

    controller = ZOnlyController()
    n_params = controller.n_params

    print(f"Z-only controller with hidden layer")
    print(f"  Architecture: z(64) -> Linear(40) -> tanh -> Linear(1)")
    print(f"  Controller params: {n_params}")
    print(f"CMA-ES: popsize={args.popsize}, σ={args.sigma}, gens={args.gens}")
    print(f"Rollouts per eval: {args.rollouts}")
    print(f"Workers: {args.workers}")
    print(f"Temperature: {TEMPERATURE}")

    # Sanity check
    print("\nSanity check...")
    rnn = MDNRNN(z_size=64, n_mix=5, rnn_size=512)
    rnn.load_state_dict(torch.load(rnn_path, map_location="cpu"))
    rnn.eval()
    init_z = np.load(init_z_path)
    init_z_dict = {"mu": init_z["mu"], "logvar": init_z["logvar"]}

    test_ctrl = ZOnlyController()
    test_reward = dream_rollout(rnn, test_ctrl, init_z_dict, torch.device("cpu"))
    print(f"  Zero-weight controller survived {test_reward} steps")
    del rnn

    # CMA-ES
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

            es.tell(solutions, [-r for r in rewards])

            best = max(rewards)
            mean = np.mean(rewards)
            worst = min(rewards)

            if best > best_ever:
                best_ever = best
                np.save(
                    os.path.join(args.output, "controller_zonly_best.npy"),
                    es.result.xbest,
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

            if gen % 25 == 0:
                np.save(
                    os.path.join(args.output, "controller_zonly_params.npy"),
                    es.result.xbest,
                )

    finally:
        pool.close()
        pool.join()

    np.save(os.path.join(args.output, "controller_zonly_params.npy"), es.result.xbest)
    print(f"\nTraining complete! Best-ever: {best_ever:.1f}")
