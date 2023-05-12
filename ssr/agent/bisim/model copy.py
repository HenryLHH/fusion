
from typing import Dict, List, Union

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override, PublicAPI

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict, TensorType

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable

from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.torch_utils import convert_to_torch_tensor, explained_variance, sequence_mask


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps


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
            nn.Linear(256, z_dim*2),             # B, z_dim*2        
        )
    
    return encoder


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()


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

class BetaVAE_H(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, z_dim=10, nc=3):
        super(BetaVAE_H, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.encoder = build_encoder_net(z_dim, nc)
        self.decoder = build_decoder_net(z_dim, nc)
        # self.predictor = nn.Sequential(
        #     nn.Linear(self.z_dim, self.z_dim*2),
        #     nn.ELU(),
        #     nn.Linear(self.z_dim*2, 18), # 3*3
        # )
        self.weight_init()
        # self.reg_loss = nn.MSELoss()
        print(self)
    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z)

        return x_recon, mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)


class TorchCustomModel(TorchModelV2, nn.Module):
    """Example of a PyTorch custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.torch_sub_model = BetaVAE_H(z_dim=32, nc=3)
        self.torch_sub_model.load_state_dict(torch.load('/home/haohong/0_MTSRL/baselines_clean/VAE/ckpt/ckpt_archive/32_latent_1000000', map_location="cpu")['model_states']['net'])
        
        self.fc_act = nn.Sequential(*[
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 4)])

        self.fc_val = nn.Sequential(*[
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)])
        
        self._feat = None
        
        p = list(self.torch_sub_model.parameters())        
        for v in p:
            v.requires_grad = False
        
    def forward(self, input_dict, state, seq_lens):
        inputs = input_dict["obs"]['img'].float()     
        inputs_recon, mu, logvar = self.torch_sub_model(inputs)
        self._feat = mu
        logits_out = self.fc_act(mu)
        self.cur_val = self.fc_val(mu)
        return logits_out, state

    def value_function(self):
        return torch.reshape(self.cur_val, [-1])

    
from ray.rllib.models.modelv2 import restore_original_dimensions
class CostValueNetwork(TorchCustomModel):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(CostValueNetwork, self).__init__(obs_space, action_space, num_outputs, model_config, name)   
        
        self.fc_cost_val = nn.Sequential(*[
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        ])
        
        self._last_cost_value = None
        # print('cost network: ', self)
        
    def forward(self, input_dict, state, seq_lens):
        print('true state in: ', input_dict.keys())
        
        if 'obs' in input_dict.keys() and 'new_obs' in input_dict.keys():
            new_obs = restore_original_dimensions(input_dict['new_obs'], self.obs_space, 'torch')        
            print('obs length: ', len(input_dict['obs']['state']))
            print('obs: ', input_dict['obs'].keys())
            print('new: ', new_obs)
            try:
                print('diff: ', torch.square(input_dict['obs']['state']-new_obs['state']).sum())
            except: 
                print('lollllllll')
                
                if isinstance(input_dict, dict):
                    if 'state' in input_dict['obs']:
                        print(input_dict['obs']['state'], input_dict['new_obs']['state'])
        ret = super(CostValueNetwork, self).forward(input_dict, state, seq_lens)
        self._last_cost_value = self.fc_cost_val(self._feat)
        return ret
    
    def get_cost_value(self):
        return self._last_cost_value.reshape(-1)

class CostValueNetworkMixin:
    def __init__(self, obs_space, action_space, config):
        if config.get("use_gae"):

            def cost_value(ob, prev_action, prev_reward, *state):
                model_out, _ = self.model(
                    {
                        SampleBatch.CUR_OBS: convert_to_torch_tensor(np.asarray([ob]), self.device),
                        SampleBatch.PREV_ACTIONS: convert_to_torch_tensor(np.asarray([prev_action]), self.device),
                        SampleBatch.PREV_REWARDS: convert_to_torch_tensor(np.asarray([prev_reward]), self.device),
                        "is_training": False,
                    }, [convert_to_torch_tensor(np.asarray([s]), self.device) for s in state], convert_to_torch_tensor(np.asarray([1]), self.device)
                )
                return self.model.get_cost_value()[0]
        else:

            def cost_value(ob, prev_action, prev_reward, *state):
                return 0.0

        self._cost_value = cost_value

from ray.rllib.models import ModelCatalog
ModelCatalog.register_custom_model(
        "my_model", TorchCustomModel
    )