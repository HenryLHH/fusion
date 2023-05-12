
from typing import Dict, List, Union

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override, PublicAPI

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import (
    normc_initializer,
    same_padding,
    SlimConv2d,
    SlimFC,
)
from ray.rllib.models.utils import get_activation_fn, get_filter_config
from ray.rllib.models.modelv2 import restore_original_dimensions

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

        
        activation = self.model_config.get("conv_activation")
        filters = [
            [16, [8, 8], 4],
            [32, [4, 4], 2],
            [256, [11, 11], 1],
        ]

        # Post FC net config.
        post_fcnet_hiddens = model_config.get("post_fcnet_hiddens", [])
        post_fcnet_activation = get_activation_fn(
            model_config.get("post_fcnet_activation"), framework="torch"
        )

        no_final_linear = self.model_config.get("no_final_linear")
        vf_share_layers = self.model_config.get("vf_share_layers")

        # Whether the last layer is the output of a Flattened (rather than
        # a n x (1,1) Conv2D).
        self.last_layer_is_flattened = False
        self._logits = None

        layers = []

        (in_channels, w, h) = obs_space.original_space['img'].shape

        in_size = [w, h]
        for out_channels, kernel, stride in filters[:-1]:
            padding, out_size = same_padding(in_size, kernel, stride)
            layers.append(
                SlimConv2d(
                    in_channels,
                    out_channels,
                    kernel,
                    stride,
                    padding,
                    activation_fn=activation,
                )
            )
            in_channels = out_channels
            in_size = out_size

        out_channels, kernel, stride = filters[-1]
        layers.append(
            SlimConv2d(
                in_channels,
                out_channels,
                kernel,
                stride,
                None,  # padding=valid
                activation_fn=activation,
            )
        )
        
        # num_outputs defined. Use that to create an exact
        # `num_output`-sized (1,1)-Conv2D.
        if num_outputs: 
            in_size = [
                np.ceil((in_size[0] - kernel[0]) / stride),
                np.ceil((in_size[1] - kernel[1]) / stride),
            ]
            padding, _ = same_padding(in_size, [1, 1], [1, 1])

            self._logits = SlimConv2d(
                out_channels,
                num_outputs,
                [1, 1],
                1,
                padding,
                activation_fn=None,
            )   
        self._convs = nn.Sequential(*layers)
        self._value_branch_separate = self._value_branch = None
        if vf_share_layers:
            self._value_branch = SlimFC(
                out_channels, 1, initializer=normc_initializer(0.01), activation_fn=None
            )
        else:
            vf_layers = []
            (in_channels, w, h) = obs_space.original_space['img'].shape
            in_size = [w, h]
            for out_channels, kernel, stride in filters[:-1]:
                padding, out_size = same_padding(in_size, kernel, stride)
                vf_layers.append(
                    SlimConv2d(
                        in_channels,
                        out_channels,
                        kernel,
                        stride,
                        padding,
                        activation_fn=activation,
                    )
                )
                in_channels = out_channels
                in_size = out_size

            out_channels, kernel, stride = filters[-1]
            vf_layers.append(
                SlimConv2d(
                    in_channels,
                    out_channels,
                    kernel,
                    stride,
                    None,
                    activation_fn=activation,
                )
            )

            vf_layers.append(
                SlimConv2d(
                    in_channels=out_channels,
                    out_channels=1,
                    kernel=1,
                    stride=1,
                    padding=None,
                    activation_fn=None,
                )
            )
            self._value_branch_separate = nn.Sequential(*vf_layers)

        # Holds the current "base" output (before logits layer).
        self._features = None

    def forward(self, input_dict, state, seq_lens):
        input_dict = restore_original_dimensions(input_dict, self.obs_space, 'torch')
        self._features = input_dict["obs"]['img'].float()     
        conv_out = self._convs(self._features)
        if not self._value_branch_separate:
            self._features = conv_out
        if self._logits:
            conv_out = self._logits(conv_out)
        if len(conv_out.shape) == 4:
            if conv_out.shape[2] != 1 or conv_out.shape[3] != 1:
                raise ValueError(
                    "Given `conv_filters` ({}) do not result in a [B, {} "
                    "(`num_outputs`), 1, 1] shape (but in {})! Please "
                    "adjust your Conv2D stack such that the last 2 dims "
                    "are both 1.".format(
                        self.model_config["conv_filters"],
                        self.num_outputs,
                        list(conv_out.shape),
                    )
                )
            logits = conv_out.squeeze(3)
            logits = logits.squeeze(2)
        
        return logits, state


    def value_function(self):
        if self._value_branch_separate:
            value = self._value_branch_separate(self._features)
            value = value.squeeze(3)
            value = value.squeeze(2)
            return value.squeeze(1)
        else:
            if not self.last_layer_is_flattened:
                features = self._features.squeeze(3)
                features = features.squeeze(2)
            else:
                features = self._features
            return self._value_branch(features).squeeze(1)

    
class CostValueNetwork(TorchCustomModel):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(CostValueNetwork, self).__init__(obs_space, action_space, num_outputs, model_config, name)   
        filters = [
            [16, [8, 8], 4],
            [32, [4, 4], 2],
            [256, [11, 11], 1],
        ]
        cost_value_network = []
        (in_channels, w, h) = obs_space.original_space['img'].shape
        in_size = [w, h]
        for out_channels, kernel, stride in filters[:-1]:
            padding, out_size = same_padding(in_size, kernel, [stride, stride])
            cost_value_network.append(
                SlimConv2d(
                    in_channels, 
                    out_channels, 
                    kernel,
                    stride,
                    padding, 
                    activation_fn=nn.Tanh
                )
            )
            in_channels = out_channels
            in_size = out_size
        out_channels, kernel, stride = filters[-1]
        cost_value_network.append(
            SlimConv2d(
                in_channels, 
                out_channels, 
                kernel, 
                stride,
                None,
                activation_fn=nn.Tanh))
        cost_value_network.append(nn.Flatten())
        cost_value_network.append(
            SlimFC(
                out_channels, 
                1, 
                initializer=normc_initializer(0.01), 
                activation_fn=None
        ))
        
        self.cost_value_network = nn.Sequential(*cost_value_network)
        self._last_cost_value = None

    def forward(self, input_dict, state, seq_lens):
        input_dict = restore_original_dimensions(input_dict, self.obs_space, 'torch')
        ret = super(CostValueNetwork, self).forward(input_dict, state, seq_lens)
        inputs = input_dict['obs']['img']
        self._last_cost_value = self.cost_value_network(inputs)

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