import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from ssr.agent.icil.mine_network import MineNetwork
from ssr.agent.icil.model import MnistEnergyNN

def CUDA(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.cuda()

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def flatten(_list):
    return [item for sublist in _list for item in sublist]

def reparameterize(mu, logstd):
    std = torch.exp(2*logstd)
    eps = torch.randn_like(std)
    return mu + eps * std


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


def build_encoder_net(z_dim, nc):

    encoder = nn.Sequential(
            nn.Conv2d(nc, 16, kernel_size=8, stride=4, padding=0), # B, 16, 20, 20
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0), # B, 32, 9, 9
            nn.ReLU(),
            nn.Conv2d(32, 256, kernel_size=11, stride=1, padding=1), # B, 256, 2, 2
            # nn.Conv2d(64, 64, 4, 1), # B, 64,  4, 4
            nn.Tanh(),
            View((-1, 256)), 
            nn.Linear(256, z_dim),             # B, z_dim*2        
        )
    
    return encoder

def build_decoder_net(z_dim, nc):
    
    decoder = nn.Sequential(
            nn.Linear(z_dim, 256),               # B, 256
            View((-1, 256, 1, 1)),               # B, 256,  1,  1
            nn.Tanh(),
            nn.ConvTranspose2d(256, 32, 11, 1, 1),      # B,  64,  8, 8
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, 2, 0),      # B,  64,  8, 8
            nn.ReLU(True),
            nn.ConvTranspose2d(16, nc, 8, 4, 0),  # B, nc, 64, 64
        )
    
    return decoder

class StudentNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, width):
        super(StudentNetwork, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.width = width

        self.layers = nn.Sequential(
            nn.Linear(in_dim, width),
            nn.ELU(),
            nn.Linear(width, width),
            nn.ELU(),
            nn.Linear(width, out_dim*2),
            nn.Tanh()
        )

    def forward(self, x):
        return self.layers(x)

class FeaturesEncoder(nn.Module):
    def __init__(self, input_size, representation_size, width):
        super().__init__()

        self.layers = build_encoder_net(z_dim=representation_size, nc=input_size)
        
    def forward(self, x):
        return self.layers(x)


class FeaturesDecoder(nn.Module):
    def __init__(self, action_size, representation_size, width):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(representation_size + action_size, width),
            nn.ELU(),
            nn.Linear(width, width),
            nn.ELU(),
            nn.Linear(width, representation_size),
        )
    
    def forward(self, x, a):
        input = torch.cat((x, a), dim=-1)
        return self.layers(input)


class ObservationsDecoder(nn.Module):
    def __init__(self, representation_size, out_size, width):
        super().__init__()

        # self.layers = nn.Sequential(
        #     nn.Linear(representation_size * 2, width),
        #     nn.ELU(),
        #     nn.Linear(width, width),
        #     nn.ELU(),
        #     nn.Linear(width, out_size),
        # ).cuda()
        self.layers = build_decoder_net(representation_size*2, nc=out_size)

    def forward(self, x, y):
        input = torch.cat((x, y), dim=-1)
        return self.layers(input)


class EnvDiscriminator(nn.Module):
    def __init__(self, representation_size, num_envs, width):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(representation_size, width),
            nn.ELU(),
            nn.Linear(width, width),
            nn.ELU(),
            nn.Linear(width, num_envs),
        )
    
    def forward(self, state):
        return self.layers(state)


class ICIL(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim_input, hidden_dim, lr=1e-4):
        super(ICIL, self).__init__()

        self.causal_feature_encoder = FeaturesEncoder(input_size=state_dim, representation_size=hidden_dim_input, width=hidden_dim)
        self.causal_feature_decoder = FeaturesDecoder(action_size=action_dim, representation_size=hidden_dim_input, width=hidden_dim)
        self.policy_network = StudentNetwork(in_dim=hidden_dim, out_dim=action_dim, width=hidden_dim)
        
        self.observation_decoder = ObservationsDecoder(representation_size=hidden_dim_input, out_size=state_dim, width=hidden_dim)

        self.noise_features_encoders = FeaturesEncoder(input_size=state_dim, representation_size=hidden_dim_input, width=hidden_dim)
        self.noise_features_decoders = FeaturesDecoder(action_size=action_dim, representation_size=hidden_dim_input, width=hidden_dim)

        self.mine_network = MineNetwork(x_dim=hidden_dim_input, z_dim=hidden_dim_input, width=hidden_dim)
        self.energy_model = MnistEnergyNN()
        self.energy_model.load_state_dict(torch.load('/home/haohong/0_causal_drive/ssr-rl/ssr/agent/icil/generative_ebm/checkpoint_12000.pt', map_location="cpu"))
        
        self.device = "cuda"

        noise_models_params = list(self.noise_features_encoders.parameters())

        self.rep_opt = optim.Adam(
            list(self.causal_feature_encoder.parameters())
            + list(self.causal_feature_decoder.parameters())
            + list(self.observation_decoder.parameters())
            + noise_models_params,
            lr=lr
        )

        self.mine_opt = optim.Adam(self.mine_network.parameters(), lr=lr)
        self.policy_opt = optim.Adam(
            list(self.causal_feature_encoder.parameters()) + 
            list(self.policy_network.parameters()), 
            lr=lr
        )
        # self.pure_opt = optim.Adam(list(self.causal_feature_encoder.parameters())
        #     + list(self.causal_feature_decoder.parameters())
        #     + list(self.observation_decoder.parameters()), lr=lr)
    
    def forward(self, s, a, label, deterministic=False):
        # print(env_ids)
        causal_rep = self.causal_feature_encoder(s)
        noise_rep = self.noise_features_encoders(s)
        # print(noise_rep.shape, a.shape)
        next_state_noise_rep = self.noise_features_decoders(noise_rep, a)
        next_state_causal_rep = self.causal_feature_decoder(causal_rep, a)


        act_logits = self.policy_network(causal_rep)
        if not deterministic: 
            act_pred = reparameterize(act_logits[:, :2], act_logits[:, 2:])
        else:
            act_pred = act_logits[:, :2]
        
        policy_loss = nn.MSELoss()(act_pred, a)
        
        predicted_next_state = self.observation_decoder(next_state_causal_rep, next_state_noise_rep)
        # print(predicted_next_state.shape, label.shape)
        next_state_pred_loss = nn.MSELoss()(predicted_next_state, label) #TODO: label = next state
        
        mi_loss = self.mine_network.mi(causal_rep, noise_rep)
        mine_loss = self.mine_network.forward(causal_rep.detach(), noise_rep.detach())
        # print(s, a, predicted_next_state)
        
        next_state_causal_rep_energy = self.causal_feature_decoder(causal_rep, act_pred)
        next_state_noise_rep_energy = self.noise_features_decoders(noise_rep, act_pred)
        
        predicted_next_state_energy = self.observation_decoder(
                next_state_causal_rep_energy, next_state_noise_rep_energy
            )
        
        next_state_energy_loss = 0.05*self.energy_model.forward(predicted_next_state_energy).mean()
        
        return policy_loss, next_state_pred_loss, next_state_energy_loss, mi_loss, mine_loss

if __name__ == '__main__': 
    model = CUDA(ICIL(state_dim=5, action_dim=2, hidden_dim_input=64, hidden_dim=64))
    print(model)
    s = CUDA(torch.rand(1, 5, 84, 84))
    s_next = CUDA(torch.rand(1, 5, 84, 84))
    
    a = CUDA(torch.rand(1, 2))
    print(model.forward(s, a, s_next))