import os
import sys

sys.path.append("..")

import argparse
import multiprocessing
import time

import cma
import numpy as np
import torch

from models.mdn_rnn import MDNRNN
from models.vae import ConvVAE

_WORKER_MDN_RNN = None
_WORKER_Z_BANK = None
_WORKER_DEVICE = None


class ControllerVizDoom:
    def __init__(self, latent_dim=64, hidden_dim=512, weights=None):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.input_dim = 1088

        if latent_dim + hidden_dim + hidden_dim != self.input_dim:
            raise ValueError("latent_dim + hidden_dim + hidden_dim must equal 1088")

        self.z = np.zeros(self.latent_dim, dtype=np.float32)
        self.h = np.zeros(self.hidden_dim, dtype=np.float32)
        self.c = np.zeros(self.hidden_dim, dtype=np.float32)

        if weights is None:
            self.weights = np.zeros(self.input_dim, dtype=np.float32)
        else:
            w = np.asarray(weights, dtype=np.float32).reshape(-1)
            if w.shape[0] != self.input_dim:
                raise ValueError(f"weights must have shape ({self.input_dim},)")
            self.weights = w

    @staticmethod
    def _to_numpy_cpu(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().reshape(-1)
        return np.asarray(x, dtype=np.float32).reshape(-1)

    def forward(self, z, h, c):
        self.z = self._to_numpy_cpu(z).astype(np.float32)
        self.h = self._to_numpy_cpu(h).astype(np.float32)
        self.c = self._to_numpy_cpu(c).astype(np.float32)

        x = np.concatenate([self.z, self.h, self.c], axis=0)
        if x.shape[0] != self.input_dim:
            raise ValueError(
                f"Concatenated input must be {self.input_dim}, got {x.shape[0]}"
            )

        out = float(np.dot(self.weights, x))
        if out < -0.33:
            return 0
        if out > 0.33:
            return 2
        return 1


def dream_rollout(
    mdn_rnn, controller, z_bank, device, temperature=1.15, max_steps=2100
):
    mdn_rnn.eval()

    with torch.no_grad():
        if isinstance(z_bank, torch.Tensor):
            start_idx = np.random.randint(0, z_bank.shape[0])
            z = z_bank[start_idx].to(device=device, dtype=torch.float32).reshape(-1)
        else:
            z_arr = np.asarray(z_bank, dtype=np.float32)
            start_idx = np.random.randint(0, z_arr.shape[0])
            z = (
                torch.from_numpy(z_arr[start_idx])
                .to(device=device, dtype=torch.float32)
                .reshape(-1)
            )

        hidden = (
            torch.zeros(1, 1, mdn_rnn.hidden_dim, device=device),
            torch.zeros(1, 1, mdn_rnn.hidden_dim, device=device),
        )

        for step in range(max_steps):
            h_cpu = hidden[0].squeeze(0).squeeze(0).detach().cpu()
            c_cpu = hidden[1].squeeze(0).squeeze(0).detach().cpu()
            action_id = controller.forward(z.detach().cpu(), h_cpu, c_cpu)

            z_in = z.view(1, 1, -1)
            action_in = torch.tensor([[action_id]], device=device)
            mdn_out, death_logits, hidden = mdn_rnn(z_in, action_in, hidden)

            pi, mu, sigma = mdn_rnn.get_mdn_params(mdn_out)
            z_next = mdn_rnn.sample(pi, mu, sigma * temperature).squeeze(0).squeeze(0)

            death_prob = torch.sigmoid(death_logits[:, -1, :]).item()
            z = z_next
            if death_prob > 0.5:
                return step + 1

        return max_steps


def init_worker(
    mdn_rnn_ckpt_path,
    z_bank_path,
    device,
    latent_dim=64,
    action_dim=3,
    hidden_dim=512,
    n_gaussians=5,
):
    global _WORKER_MDN_RNN, _WORKER_Z_BANK, _WORKER_DEVICE

    if str(device).startswith("cuda") and not torch.cuda.is_available():
        _WORKER_DEVICE = torch.device("cpu")
    else:
        _WORKER_DEVICE = torch.device(device)

    # Build and load a worker-local MDN-RNN copy.
    _WORKER_MDN_RNN = MDNRNN(
        latent_dim=latent_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        n_gaussians=n_gaussians,
    ).to(_WORKER_DEVICE)

    ckpt = torch.load(mdn_rnn_ckpt_path, map_location=_WORKER_DEVICE)
    state_dict = (
        ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    )
    _WORKER_MDN_RNN.load_state_dict(state_dict)
    _WORKER_MDN_RNN.eval()

    # Load worker-local latent bank copy.
    if z_bank_path.endswith(".pt"):
        z_bank_obj = torch.load(z_bank_path, map_location="cpu")
        if isinstance(z_bank_obj, torch.Tensor):
            z_bank_arr = z_bank_obj.detach().cpu().numpy()
        else:
            z_bank_arr = np.asarray(z_bank_obj, dtype=np.float32)
    elif z_bank_path.endswith(".npz"):
        z_bank_npz = np.load(z_bank_path)
        first_key = list(z_bank_npz.keys())[0]
        z_bank_arr = z_bank_npz[first_key]
    else:
        z_bank_arr = np.load(z_bank_path)

    _WORKER_Z_BANK = torch.as_tensor(
        z_bank_arr, dtype=torch.float32, device=_WORKER_DEVICE
    )


def evaluate_worker(args):
    weights, num_rollouts, temperature, max_steps = args

    if _WORKER_MDN_RNN is None or _WORKER_Z_BANK is None or _WORKER_DEVICE is None:
        raise RuntimeError("Worker not initialized. Call init_worker first.")

    controller = ControllerVizDoom(weights=weights)
    rewards = []
    for _ in range(num_rollouts):
        reward = dream_rollout(
            _WORKER_MDN_RNN,
            controller,
            _WORKER_Z_BANK,
            _WORKER_DEVICE,
            temperature=temperature,
            max_steps=max_steps,
        )
        rewards.append(reward)

    return float(np.mean(rewards))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_path", type=str, default="../outputs/vizdoom/vae.pth")
    parser.add_argument(
        "--mdn_path", type=str, default="../outputs/vizdoom/mdn_rnn.pth"
    )
    parser.add_argument(
        "--frames_path", type=str, default="../outputs/vizdoom/frames.npy"
    )
    parser.add_argument("--output_dir", type=str, default="../outputs/vizdoom")
    parser.add_argument("--gens", type=int, default=200)
    parser.add_argument("--popsize", type=int, default=64)
    parser.add_argument("--rollouts", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.15)
    parser.add_argument("--max_steps", type=int, default=2100)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--z_bank_size", type=int, default=5000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # === Phase 1: Build z bank from real frames ===
    print("Building z bank from real frames...")
    vae = ConvVAE(latent_dim=64)
    vae.load_state_dict(torch.load(args.vae_path, map_location=device))
    vae.to(device)
    vae.eval()

    frames = np.load(args.frames_path)
    # Sample a subset if we have more than z_bank_size
    if len(frames) > args.z_bank_size:
        indices = np.random.choice(len(frames), args.z_bank_size, replace=False)
        frames = frames[indices]

    frames_tensor = torch.from_numpy(frames).float() / 255.0
    frames_tensor = frames_tensor.permute(0, 3, 1, 2)

    all_z = []
    with torch.no_grad():
        for i in range(0, len(frames_tensor), 256):
            batch = frames_tensor[i : i + 256].to(device)
            mu, _ = vae.encode(batch)
            all_z.append(mu.cpu())

    z_bank = torch.cat(all_z, dim=0)
    z_bank_path = os.path.join(args.output_dir, "z_bank.npy")
    np.save(z_bank_path, z_bank.numpy())

    print(f"z bank: {z_bank.shape} saved to {z_bank_path}")

    # VAE no longer needed
    del vae, frames, frames_tensor, all_z
    # === Phase 2: CMA-ES Evolution ===
    n_params = 1088
    print(
        f"\nStarting CMA-ES: {n_params} params, pop={args.popsize}, "
        f"τ={args.temperature}, rollouts={args.rollouts}"
    )

    es = cma.CMAEvolutionStrategy(
        n_params * [0],
        0.5,
        {"popsize": args.popsize},
    )

    print(f"Initializing {args.workers} workers...")
    pool = multiprocessing.Pool(
        processes=args.workers,
        initializer=init_worker,
        initargs=(args.mdn_path, z_bank_path, "cpu"),
    )

    history = {"generation": [], "best": [], "mean": [], "worst": []}

    try:
        for gen in range(args.gens):
            t0 = time.time()

            solutions = es.ask()
            eval_args = [
                (weights, args.rollouts, args.temperature, args.max_steps)
                for weights in solutions
            ]
            rewards = pool.map(evaluate_worker, eval_args)

            # CMA-ES minimizes, so negate
            es.tell(solutions, [-r for r in rewards])

            best = max(rewards)
            mean = np.mean(rewards)
            worst = min(rewards)

            history["generation"].append(gen)
            history["best"].append(best)
            history["mean"].append(mean)
            history["worst"].append(worst)

            elapsed = time.time() - t0
            print(
                f"Gen {gen:3d} | Best: {best:7.1f} | Mean: {mean:7.1f} | "
                f"Worst: {worst:7.1f} | Time: {elapsed:.1f}s"
            )

            # Checkpoint every 25 generations
            if gen % 25 == 0:
                np.save(
                    os.path.join(args.output_dir, "controller_params.npy"),
                    es.result.xbest,
                )

    finally:
        pool.close()
        pool.join()

    # Save final results
    np.save(os.path.join(args.output_dir, "controller_params.npy"), es.result.xbest)
    np.save(os.path.join(args.output_dir, "training_history.npy"), history)

    print(f"\nTraining complete!")
    print(f"Best survival time: {max(history['best']):.1f} steps")
    print(f"Final mean: {history['mean'][-1]:.1f} steps")
