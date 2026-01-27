import torch
import numpy as np
import cma
import gymnasium as gym
from vae import ConvVAE
from mdn_rnn import MDNRNN
from controller import Controller

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load trained models
vae = ConvVAE(latent_dim=32)
vae.load_state_dict(torch.load("outputs/vae.pth"))
vae.to(device)
vae.eval()

mdn_rnn = MDNRNN(latent_dim=32, action_dim=3, hidden_dim=256, n_gaussians=5)
mdn_rnn.load_state_dict(torch.load("outputs/mdn_rnn.pth"))
mdn_rnn.to(device)
mdn_rnn.eval()

controller = Controller()
controller.to(device)


def get_action(obs, hidden, controller, vae, mdn_rnn):
    # Get action from obervation using V, M , C
    with torch.no_grad():
        # Obeservation to z
        obs_tensor = torch.from_numpy(obs).float() / 255.0
        obs_tensor = obs_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
        mu, _ = vae.encode(obs_tensor)
        z = mu.squeeze(0)

        # Hidden state from RNN
        h = hidden[0].squeeze(0).squeeze(0)

        # Action from controller
        action = controller(z, h)

        return action.cpu().numpy(), z


def rollout(controller, vae, mdn_rnn, render=False):
    # Run one episode, return total reward.
    env = gym.make("CarRacing-v3", continuous=True)
    obs, _ = env.reset()

    # Resize obs
    from PIL import Image

    obs = np.array(Image.fromarray(obs).resize((64, 64)))

    # Initialize RNN hidden state
    hidden = (torch.zeros(1, 1, 256).to(device), torch.zeros(1, 1, 256).to(device))
    total_reward = 0

    for step in range(500):
        action, z = get_action(obs, hidden, controller, vae, mdn_rnn)

        # Gas and brakes
        action_env = np.array([action[0], (action[1] + 1) / 2, (action[2] + 1) / 2])
        obs, reward, terminated, truncated, _ = env.step(action_env)
        obs = np.array(Image.fromarray(obs).resize((64, 64)))
        total_reward += reward

        # Update RNN hidden
        with torch.no_grad():
            z_input = z.unsqueeze(0).unsqueeze(0)
            a_input = (
                torch.from_numpy(action).float().unsqueeze(0).unsqueeze(0).to(device)
            )
            _, hidden = mdn_rnn(z_input, a_input, hidden)

        if terminated or truncated:
            break

    env.close()
    return total_reward


def set_controller_params(contoller, params):
    # Set controller weights from flat param vector
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


if __name__ == "__main__":
    # CMA-ES optimization
    n_params = sum(p.numel() for p in controller.parameters())
    print(f"Optimizing {n_params} parameters")

    es = cma.CMAEvolutionStrategy(n_params * [0], 0.5)

    generation = 0
    while not es.stop():
        # Get population of candidate solutions
        solutions = es.ask()

        fitness = []
        for params in solutions:
            set_controller_params(controller, np.array(params))
            reward = rollout(controller, vae, mdn_rnn)
            fitness.append(-reward)

        es.tell(solutions, fitness)

        generation += 1
        best_reward = -min(fitness)
        mean_reward = -np.mean(fitness)
        print(f"Gen {generation}: Best={best_reward:.1f}, Mean={mean_reward:.1f}")

        if generation >= 50:
            break

    # Save best controller as raw numpy array
    best_params = np.array(es.result.xbest)
    np.save("outputs/controller_params.npy", best_params)
    print(f"Saved best params to outputs/controller_params.npy")

    # Verify it works
    set_controller_params(controller, best_params)
    test_reward = rollout(controller, vae, mdn_rnn)
    print(f"Verification reward: {test_reward:.1f}")

    # Also verify load works
    loaded_params = np.load("outputs/controller_params.npy")
    set_controller_params(controller, loaded_params)
    test_reward2 = rollout(controller, vae, mdn_rnn)
    print(f"After reload reward: {test_reward2:.1f}")
