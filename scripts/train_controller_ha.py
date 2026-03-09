#!/usr/bin/env python3
"""
CMA-ES controller training matching Ha's implementation exactly.

Key differences from our previous attempts:
1. Controller outputs continuous action (tanh) not discrete
2. Continuous action fed directly to RNN (not discretized)
3. Full [z, c, h] input (1088 params) - Ha's default
4. Also supports z-only with hidden layer (2641 params)

Usage:
    python train_controller_ha.py --data_dir ../outputs/vizdoom/ha_exact --mode full
    python train_controller_ha.py --data_dir ../outputs/vizdoom/ha_exact --mode zonly
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
# Controllers
# ================================================================


class FullController:
    """Ha's controller: W·[z,c,h] -> tanh -> continuous action.
    1088 params.
    """

    def __init__(self, weights=None):
        self.n_params = 64 + 512 + 512  # z + c + h = 1088
        if weights is None:
            self.weights = np.zeros(self.n_params, dtype=np.float32)
        else:
            self.weights = np.asarray(weights, dtype=np.float32).flatten()

    def act(self, z, h=None, c=None):
        obs = np.concatenate([z, c, h])
        out = float(np.dot(self.weights, obs))
        return np.tanh(out)  # continuous in [-1, 1]


class ZOnlyController:
    """z(64) -> Linear(40) -> tanh -> Linear(1) -> tanh -> continuous action.
    2641 params.
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

    def act(self, z, h=None, c=None):
        hidden = np.tanh(self.W1 @ z + self.b1)
        out = float((self.W2 @ hidden + self.b2)[0])
        return np.tanh(out)  # continuous in [-1, 1]


# ================================================================
# Dream rollout with CONTINUOUS actions
# ================================================================

TEMPERATURE = 1.15
MAX_FRAMES = 2100


def dream_rollout(rnn, controller, initial_z_data, device, use_rnn_obs):
    """Dream rollout matching Ha's DoomCoverRNNEnv._step exactly.

    Key: controller outputs continuous action, which is fed directly
    to the RNN. No discretization.
    """
    rnn.eval()

    with torch.no_grad():
        idx = np.random.randint(0, len(initial_z_data["mu"]))
        init_mu = initial_z_data["mu"][idx]
        init_logvar = initial_z_data["logvar"][idx]
        z = init_mu + np.exp(init_logvar / 2.0) * np.random.randn(*init_logvar.shape)

        state = rnn.init_state(1, device)
        restart = 1.0

        for step in range(MAX_FRAMES):
            if use_rnn_obs:
                h_np = state[0][0].cpu().numpy()
                c_np = state[1][0].cpu().numpy()
                action = controller.act(z, h_np, c_np)  # continuous [-1, 1]
            else:
                action = controller.act(z)  # z-only, continuous [-1, 1]

            # Feed CONTINUOUS action to RNN (this is the key difference)
            z_tensor = torch.from_numpy(z).float().unsqueeze(0).to(device)
            logmix, mean, logstd, restart_logit, state = rnn.forward_step(
                z_tensor, action, restart, state  # action is continuous float!
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
_WORKER_MODE = None


def init_worker(rnn_path, init_z_path, device_str, mode):
    global _WORKER_RNN, _WORKER_INIT_Z, _WORKER_DEVICE, _WORKER_MODE

    _WORKER_DEVICE = torch.device(device_str)
    _WORKER_MODE = mode
    _WORKER_RNN = MDNRNN(z_size=64, n_mix=5, rnn_size=512).to(_WORKER_DEVICE)
    ckpt = torch.load(rnn_path, map_location=_WORKER_DEVICE)
    _WORKER_RNN.load_state_dict(ckpt)
    _WORKER_RNN.eval()

    data = np.load(init_z_path)
    _WORKER_INIT_Z = {"mu": data["mu"], "logvar": data["logvar"]}


def evaluate_worker(args):
    weights, num_rollouts = args

    if _WORKER_MODE == "full":
        controller = FullController(weights=weights)
        use_rnn_obs = True
    else:
        controller = ZOnlyController(weights=weights)
        use_rnn_obs = False

    rewards = []
    for _ in range(num_rollouts):
        r = dream_rollout(
            _WORKER_RNN, controller, _WORKER_INIT_Z, _WORKER_DEVICE, use_rnn_obs
        )
        rewards.append(r)

    return float(np.mean(rewards))


# ================================================================
# Main
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../outputs/vizdoom/ha_exact")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--mode", type=str, default="zonly", choices=["full", "zonly"])
    parser.add_argument("--gens", type=int, default=2000)
    parser.add_argument("--popsize", type=int, default=64)
    parser.add_argument("--rollouts", type=int, default=16)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--sigma", type=float, default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = args.data_dir
    if args.sigma is None:
        args.sigma = 0.5 if args.mode == "full" else 0.1

    import cma

    os.makedirs(args.output, exist_ok=True)

    rnn_path = os.path.join(args.data_dir, "mdn_rnn.pth")
    init_z_path = os.path.join(args.data_dir, "initial_z.npz")

    if args.mode == "full":
        controller = FullController()
        name_prefix = "controller"
    else:
        controller = ZOnlyController()
        name_prefix = "controller_zonly"

    n_params = controller.n_params

    print(f"Mode: {args.mode}")
    print(f"Controller params: {n_params}")
    print(f"CMA-ES: popsize={args.popsize}, σ={args.sigma}, gens={args.gens}")
    print(f"Rollouts per eval: {args.rollouts}")
    print(f"Workers: {args.workers}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Actions: CONTINUOUS [-1, 1]")

    # Sanity check
    print("\nSanity check...")
    rnn = MDNRNN(z_size=64, n_mix=5, rnn_size=512)
    rnn.load_state_dict(torch.load(rnn_path, map_location="cpu"))
    rnn.eval()
    init_z = np.load(init_z_path)
    init_z_dict = {"mu": init_z["mu"], "logvar": init_z["logvar"]}

    test_reward = dream_rollout(
        rnn,
        controller,
        init_z_dict,
        torch.device("cpu"),
        use_rnn_obs=(args.mode == "full"),
    )
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
        initargs=(rnn_path, init_z_path, "cpu", args.mode),
    )

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
                    os.path.join(args.output, f"{name_prefix}_best.npy"),
                    es.result.xbest,
                )

            elapsed = time.time() - t0
            print(
                f"Gen {gen:3d} | Best: {best:7.1f} | Mean: {mean:7.1f} | "
                f"Worst: {worst:7.1f} | Best-ever: {best_ever:7.1f} | {elapsed:.1f}s"
            )

            if gen % 25 == 0:
                np.save(
                    os.path.join(args.output, f"{name_prefix}_params.npy"),
                    es.result.xbest,
                )

    finally:
        pool.close()
        pool.join()

    np.save(os.path.join(args.output, f"{name_prefix}_params.npy"), es.result.xbest)
    print(f"\nTraining complete! Best-ever: {best_ever:.1f}")
