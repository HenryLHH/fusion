import numpy as np
import torch

device='cuda:0'

def initialize_replay_buffer(replay_buffer, n=32, state_dim=35):
    x = torch.rand(n, state_dim)
    for i in range(n):
        replay_buffer.append((x[i]))


def sample_from_replay_buffer(replay_buffer, n=32, state_dim=35):
    x_negative = torch.rand(n, state_dim)
    idx_samples = []
    n_samples = np.random.binomial(n, 0.95)
    idx_samples = np.random.choice(range(len(replay_buffer)),
                                   size=n_samples, replace=False)
    for i, idx in enumerate(idx_samples):
        x_negative[i] = replay_buffer[idx][0]
    return x_negative


def update_replay_buffer(replay_buffer, batch):
    x = batch
    for i in range(x.shape[0]):
        replay_buffer.append((x[i]))


def langevin_rollout(x_negative, energy_nn, step_size, lambda_var, K):
    x_negative = x_negative.requires_grad_(True)
    for k in range(K):
        energy_nn(x_negative).sum().backward()
        with torch.no_grad():
            x_negative.grad.clamp_(-0.01, 0.01)
            x_negative -= step_size * x_negative.grad / 2
            x_negative += lambda_var * torch.randn(*x_negative.shape).to(device)
            x_negative.clamp_(0, 1)
        x_negative.grad.zero_()
    return x_negative.requires_grad_(False)

def loss_fn(x_positive, x_negative, energy_nn, alpha):
    x_positive_energy = energy_nn(x_positive)
    x_negative_energy = energy_nn(x_negative)
    loss_l2 = alpha * (x_positive_energy.pow(2) + x_negative_energy.pow(2)).mean()
    loss_ml = (x_positive_energy - x_negative_energy).mean()
    return loss_l2 + loss_ml, loss_l2.item(), loss_ml.item()