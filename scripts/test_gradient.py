"""Diagnostic: check if GMM gradients are flowing."""

import torch
import sys

sys.path.insert(0, "..")

from models.mdn_rnn import MDNRNN

device = torch.device("cpu")
model = MDNRNN(latent_dim=64, action_dim=3, hidden_dim=512, n_gaussians=5).to(device)

batch, seq = 4, 50
z = torch.randn(batch, seq, 64, device=device)
action = torch.randint(0, 3, (batch, seq), device=device)
restart = torch.zeros(batch, seq, device=device)
restart[:, 0] = 1.0

target_z = torch.randn(batch, seq - 1, 64, device=device)

# Forward
output, hidden = model(z[:, :-1], action[:, :-1], restart[:, :-1])
restart_logits, logmix, mean, logstd = model.get_mdn_params(output)

print(f"output shape: {output.shape}")
print(f"logmix: {logmix.shape}, mean: {mean.shape}, logstd: {logstd.shape}")
print(f"logmix range: [{logmix.min():.4f}, {logmix.max():.4f}]")
print(f"mean range: [{mean.min():.4f}, {mean.max():.4f}]")
print(f"logstd range: [{logstd.min():.4f}, {logstd.max():.4f}]")
print()

# Compute z_cost only
z_cost = model.loss_function(logmix, mean, logstd, target_z)
print(f"z_cost: {z_cost.item():.6f}")

# Backward
z_cost.backward()

# Check fc_out gradients
fc_w = model.fc_out.weight.grad
fc_b = model.fc_out.bias.grad

print(f"\n=== fc_out.weight.grad (shape {fc_w.shape}) ===")
print(f"Row 0 (restart):  norm={fc_w[0].norm():.6f}")
print(
    f"Rows 1-960 (GMM): norm={fc_w[1:].norm():.6f}, mean={fc_w[1:].mean():.8f}, max_abs={fc_w[1:].abs().max():.6f}"
)

print(f"\n=== fc_out.bias.grad ===")
print(f"Row 0 (restart):  {fc_b[0]:.6f}")
print(
    f"Rows 1-960 (GMM): norm={fc_b[1:].norm():.6f}, mean={fc_b[1:].mean():.8f}, max_abs={fc_b[1:].abs().max():.6f}"
)

# LSTM
print(f"\n=== LSTM gradients ===")
for name, p in model.lstm.named_parameters():
    if p.grad is not None:
        print(f"  {name}: norm={p.grad.norm():.6f}")

# Verify: does a manual GMM gradient test work?
print(f"\n=== Manual gradient test ===")
test_mean = torch.zeros(1, requires_grad=True)
test_logstd = torch.zeros(1, requires_grad=True)
test_logmix = torch.tensor([0.0], requires_grad=True)
test_target = torch.tensor([1.5])
log_sqrt_2pi = 0.5 * torch.log(torch.tensor(2.0 * 3.14159265))
lp = (
    -0.5 * ((test_target - test_mean) / torch.exp(test_logstd)) ** 2
    - test_logstd
    - log_sqrt_2pi
)
loss = -(test_logmix + lp).logsumexp(dim=-1).mean()
loss.backward()
print(f"  test loss: {loss.item():.4f}")
print(f"  d(loss)/d(mean): {test_mean.grad.item():.4f}")
print(f"  d(loss)/d(logstd): {test_logstd.grad.item():.4f}")
