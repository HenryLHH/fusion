import torch
import torch.optim as optim
from torch import autograd, nn
from tqdm import tqdm

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
        sgld_buffer_size,
        sgld_learn_rate,
        sgld_noise_coef,
        sgld_num_steps,
        sgld_reinit_freq,
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

        pos_x = torch.FloatTensor(samples["state"]).to(self.device)
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
