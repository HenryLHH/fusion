import torch
import torch.nn as nn
from utils.utils import CPU, CUDA, reparameterize

OBS_DIM = 35

def build_encoder_net(z_dim, nc, deterministic=False):
    
    if deterministic: 
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
    
    else:
        encoder = nn.Sequential(
                nn.Conv2d(nc, 16, kernel_size=8, stride=4, padding=0), # B, 16, 20, 20
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0), # B, 32, 9, 9
                nn.ReLU(),
                nn.Conv2d(32, 256, kernel_size=11, stride=1, padding=1), # B, 256, 2, 2
                # nn.Conv2d(64, 64, 4, 1), # B, 64,  4, 4
                nn.Tanh(),
                View((-1, 256)), 
                nn.Linear(256, z_dim*2),             # B, z_dim*2        
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

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class Attention(nn.Module):
    def __init__(self, in_features, bias=False):
        super().__init__()
        self.M =  nn.Parameter(torch.nn.init.normal_(torch.zeros(in_features,in_features), mean=0, std=1))
        self.sigmd = torch.nn.Sigmoid()
        #self.M =  nn.Parameter(torch.zeros(in_features,in_features))
        #self.A = torch.zeros(in_features,in_features).to(device)

    def attention(self, z, e):
        a = z.matmul(self.M).matmul(e.permute(0,2,1))
        a = self.sigmd(a)
        #print(self.M)
        A = torch.softmax(a, dim = 1)
        e = torch.matmul(A,e)
        return e, A


class ImageStateEncoder(nn.Module):
    def __init__(self, hidden_dim=16, nc=3, output_dim=OBS_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.nc = nc
        self.output_dim = output_dim

        self.encoder = nn.ModuleList([build_encoder_net(hidden_dim//2, 1) for _ in range(nc)])
        self.fc_out = nn.Linear(hidden_dim*nc, output_dim*2)

    def forward(self, x): 
        z = []
        for i in range(len(self.encoder)):
            z_tmp = self.encoder[i](x[:, [i], :, :])
            z.append(z_tmp)

        z = torch.cat(z, dim=-1)
        output = self.fc_out(z)

        mu, std = output[:, :self.output_dim], output[:, self.output_dim:]
        state_pred = reparameterize(mu, std)

        return state_pred, mu, std
    
class ImageStateEncoder_Indiv(nn.Module):
    def __init__(self, hidden_dim=16, nc=3, output_dim=OBS_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.nc = nc
        self.output_dim = output_dim

        self.encoder = nn.ModuleList([build_encoder_net(hidden_dim//2, 1) for _ in range(nc)])
    
    def forward(self, x): 
        z = []
        for i in range(len(self.encoder)):
            z_tmp = self.encoder[i](x[:, [i], :, :])
            z.append(z_tmp)

        z = torch.cat(z, dim=-1)
        return z
    

class ImageStateEncoder_GRU(nn.Module):
    def __init__(self, hidden_dim=16, nc=3, output_dim=OBS_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.nc = nc
        self.output_dim = output_dim

        self.encoder = nn.ModuleList(
            [build_encoder_net(hidden_dim//2, 1) for _ in range(nc-1)])
        self.encoder_ego = build_encoder_net(output_dim, 1)

        # self.fc_out = nn.Linear(hidden_dim*nc, output_dim*2)
        self.aggregator = Attention(in_features=hidden_dim) # nn.GRU(input_size=output_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.h_0 = torch.zeros(2, hidden_dim)
        
    def forward(self, x): 
        z = []
        for idx_enc, i in enumerate([0, 4, 3, 2]): # range(len(self.encoder)):
            z_tmp = self.encoder[idx_enc](x[:, [i], :, :])
            z.append(z_tmp.unsqueeze(1))
        
        z = torch.cat(z, dim=1)
        z_ego = self.encoder_ego(x[:, [1], :, :]) # ego states
        z_ego = reparameterize(z_ego[:, :self.output_dim], torch.exp(z_ego[:, self.output_dim:]))
        z_ego = z_ego.unsqueeze(1)

        # output = self.fc_out(z)
        output, _ = self.aggregator.attention(z_ego, z)
        # mu, std = output[:, :self.output_dim], output[:, self.output_dim:]
        # state_pred = reparameterize(mu, std)
        output = output.squeeze(1)
        return output, None, None # state_pred, mu, std


class ImageStateEncoder_NonCausal(nn.Module):
    def __init__(self, hidden_dim, nc, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nc = nc
        
        self.encoder = build_encoder_net(hidden_dim*nc, nc)
        self.fc_out = nn.Linear(hidden_dim*nc, output_dim*2)
        # self.ln = nn.LayerNorm(OBS_DIM)

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x): 

        z_tmp = self.encoder(x)
        mu, std = z_tmp[:, :self.hidden_dim*self.nc], z_tmp[:, self.hidden_dim*self.nc:]
        z_tmp = reparameterize(mu, std)
        
        output = self.fc_out(z_tmp)

        mu, std = output[:, :self.output_dim], output[:, self.output_dim:]
        state_pred = reparameterize(mu, std)

        return state_pred, mu, torch.exp(std)


class StateEncoder(nn.Module):
    def __init__(self, obs_dim, hidden_dim):
        super().__init__()
        self.encoder_1 = nn.Linear(obs_dim, hidden_dim)
        self.encoder_2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):        
        x = self.encoder_1(x)
        x = torch.tanh(x)
        x = self.encoder_2(x)
        x = torch.tanh(x)
        return x