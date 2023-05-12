import copy

import numpy as np
# from ray.rllib.agents.ppo.ppo_torch_policy import SampleBatch
from ray.rllib.policy.sample_batch import SampleBatch

from ray.rllib.models.torch.misc import normc_initializer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.visionnet import VisionNetwork

from ray.rllib.utils import try_import_torch
from ray.rllib.utils.framework import get_activation_fn

from ray.rllib.utils.torch_utils import convert_to_torch_tensor, explained_variance, sequence_mask
from ray.rllib.models.torch.misc import normc_initializer, same_padding, \
    SlimConv2d, SlimFC
import gym
torch, nn = try_import_torch()


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


class CostValueNetwork(TorchModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(CostValueNetwork, self).__init__(obs_space, action_space, num_outputs, model_config, name)

        '''
        hiddens = model_config.get("fcnet_hiddens")
        activation = get_activation_fn(model_config.get("fcnet_activation"))
        input_dim = np.product(obs_space.shape)
        print(hiddens, activation)
        layer_list = [nn.Linear(input_dim, hiddens[0]), nn.Tanh()] # activation()]
        for i in range(len(hiddens)-1):
            layer_list.append(nn.Linear(hiddens[i], hiddens[i+1]))
            layer_list.append(nn.Tanh())
        
        layer_list.append(nn.Linear(hiddens[-1], 1))
        # print(layer_list)
        self.cost_value_network = nn.Sequential(*layer_list)
        '''
        activation = get_activation_fn(model_config.get("fcnet_activation"))
        filters = [
            [16, [8, 8], 4],
            [32, [4, 4], 2],
            [256, [11, 11], 1],
        ]
        cost_value_network = []
        (w, h, in_channels) = obs_space.shape
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
        ret = super(CostValueNetwork, self).forward(input_dict, state, seq_lens)
        # input_dict_copy = copy.deepcopy(input_dict)
        # input_dict_copy['obs'] = input_dict_copy['obs'][:, :, :, 1:]
        inputs = input_dict['obs'].permute(0, 3, 1, 2)
        self._last_cost_value = self.cost_value_network(inputs)
        # print('ret: ', ret, self._last_cost_value)
        # print(self._last_cost_value.shape)
        if torch.isnan(input_dict["obs"]).any():
            assert 1==2, '========================== input obs nan! ==========================='
        if torch.isnan(ret[0]).any():
            assert 1==2, '========================== return nan! ==========================='
        return ret

    def get_cost_value(self):
        return self._last_cost_value.reshape(-1)
