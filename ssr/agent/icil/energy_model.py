import torch
import torch.optim as optim
from torch import autograd, nn
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import logging

from collections import deque


from utils.dataset import Dataset_EBM
from utils.utils import CUDA
from ssr.agent.icil.icil_utils import (
    initialize_replay_buffer, 
    sample_from_replay_buffer,
    update_replay_buffer,
    langevin_rollout,
    loss_fn
)


# Fully Connected Network
def get_activation(s_act):
    if s_act == "relu":
        return nn.ReLU(inplace=True)
    elif s_act == "sigmoid":
        return nn.Sigmoid()
    elif s_act == "softplus":
        return nn.Softplus()
    elif s_act == "linear":
        return None
    elif s_act == "tanh":
        return nn.Tanh()
    elif s_act == "leakyrelu":
        return nn.LeakyReLU(0.2, inplace=True)
    elif s_act == "softmax":
        return nn.Softmax(dim=1)
    else:
        raise ValueError(f"Unexpected activation: {s_act}")


class EnergyModelNetworkMLP(nn.Module):
    """fully-connected network"""

    def __init__(self, in_dim, out_dim, l_hidden=(50,), activation="sigmoid", out_activation="linear"):
        super(EnergyModelNetworkMLP, self).__init__()
        l_neurons = tuple(l_hidden) + (out_dim,)
        if isinstance(activation, str):
            activation = (activation,) * len(l_hidden)
        activation = tuple(activation) + (out_activation,)

        l_layer = []
        prev_dim = in_dim
        for i_layer, (n_hidden, act) in enumerate(zip(l_neurons, activation)):
            l_layer.append(nn.Linear(prev_dim, n_hidden))
            act_fn = get_activation(act)
            if act_fn is not None:
                l_layer.append(act_fn)
            prev_dim = n_hidden

        self.net = nn.Sequential(*l_layer)
        self.in_dim = in_dim
        self.out_shape = (out_dim,)
        print(self.net)
    def forward(self, x):
        return self.net(x)


class EnergyModel:
    def __init__(
        self,
        in_dim,
        width,
        batch_size,
        adam_alpha,
        buffer,
        sgld_buffer_size=10000,
        sgld_learn_rate=0.01,
        sgld_noise_coef=0.01,
        sgld_num_steps=100,
        sgld_reinit_freq=0.05,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.in_dim = in_dim
        self.width = width
        self.batch_size = batch_size
        self.adam_alpha = adam_alpha

        self.buffer = buffer

        self.sgld_buffer = self._get_random_states(sgld_buffer_size)
        self.sgld_learn_rate = sgld_learn_rate
        self.sgld_noise_coef = sgld_noise_coef
        self.sgld_num_steps = sgld_num_steps
        self.sgld_reinit_freq = sgld_reinit_freq

        self.energy_network = EnergyModelNetworkMLP(
            in_dim=in_dim, out_dim=1, l_hidden=(self.width, self.width), activation="relu", out_activation="linear"
        )
        self.energy_network.to(self.device)

        self.optimizer = optim.Adam(self.energy_network.parameters(), lr=self.adam_alpha)

    def forward(self, x):
        z = self.energy_network(x)
        return z

    def train(self, num_updates):
        for update_index in tqdm(range(num_updates)):
            self._update_energy_model()
    
    def _update_energy_model(self):
        samples = self.buffer.sample()

        pos_x = torch.FloatTensor(samples).to(self.device)
        neg_x = self._sample_via_sgld()

        self.optimizer.zero_grad()
        pos_out = self.energy_network(pos_x)
        neg_out = self.energy_network(neg_x)

        contrastive_loss = (pos_out - neg_out).mean()
        reg_loss = (pos_out**2 + neg_out**2).mean()
        loss = contrastive_loss + reg_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.energy_network.parameters(), max_norm=0.1)
        self.optimizer.step()

    def _initialize_sgld(self):
        indices = torch.randint(0, len(self.sgld_buffer), (self.batch_size,))

        buffer_samples = self.sgld_buffer[indices]
        random_samples = self._get_random_states(self.batch_size)

        mask = (torch.rand(self.batch_size) < self.sgld_reinit_freq).float()[:, None]
        samples = (1 - mask) * buffer_samples + mask * random_samples

        return samples.to(self.device), indices

    def _sample_via_sgld(self) -> torch.Tensor:
        samples, indices = self._initialize_sgld()

        l_samples = []
        l_dynamics = []

        x = samples
        x.requires_grad = True

        for _ in range(self.sgld_num_steps):
            l_samples.append(x.detach().to(self.device))
            noise = torch.randn_like(x) * self.sgld_noise_coef

            out = self.energy_network(x)
            grad = autograd.grad(out.sum(), x, only_inputs=True)[0]

            dynamics = self.sgld_learn_rate * grad + noise

            x = x + dynamics
            l_samples.append(x.detach().to(self.device))
            l_dynamics.append(dynamics.detach().to(self.device))

        samples = l_samples[-1]

        self.sgld_buffer[indices] = samples.cpu()

        return samples

    def _get_random_states(self, num_states):
        return torch.FloatTensor(num_states, self.in_dim).uniform_(-1, 1)

if __name__ == '__main__': 
    device = 'cuda:0'
    K = 60
    step_size = 20
    lambda_var = 0.01
    alpha = 1

    lr = 0.001
    batch_size = 128
    replay_buffer = deque(maxlen=1000)

    energy_nn = CUDA(EnergyModelNetworkMLP(in_dim=35, out_dim=1, l_hidden=(64, 64), activation='relu', out_activation='linear'))
    optimizer = optim.Adam(energy_nn.parameters())
    initialize_replay_buffer(replay_buffer, n=batch_size)
    # data_loader = torch.utils.data.DataLoader(
    #     torchvision.datasets.MNIST('.', transform=torchvision.transforms.ToTensor(), download=True),
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=2)

    train_set = Dataset_EBM(file_path='/home/haohong/0_causal_drive/baselines_clean/envs/data_mixed_dynamics_post', \
        noise_scale=0, num_files=300000, balanced=True, image=False) # TODO: //10
    data_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=16)

    logging.basicConfig(filename='log/icil_ebm_state.log', level=logging.DEBUG)

    it = 0
    while True:
        print(it)
        x_positive = next(iter(data_loader))
        x_negative = sample_from_replay_buffer(replay_buffer, n=batch_size)
        x_positive = x_positive.to(device)
        x_negative = x_negative.to(device)

        langevin_rollout(x_negative, energy_nn, step_size, lambda_var, K)
        update_replay_buffer(replay_buffer, (x_negative.to('cpu')))
        if it % 200 == 0:
            if it > 0: 
                print(it, loss.item(), loss_ml, loss_l2)
            torch.save(energy_nn.state_dict(), 'ssr/agent/icil/ckpt_state/checkpoint_{:05d}.pt'.format(it))
        if it > 1000: 
            break
        optimizer.zero_grad()
        loss, loss_l2, loss_ml = loss_fn(x_positive, x_negative, energy_nn, alpha)
        logging.info('%f,%f' % (loss_l2, loss_ml))
        loss.backward()
        torch.nn.utils.clip_grad_value_(energy_nn.parameters(), 0.01)
        optimizer.step()

        it += 1
    