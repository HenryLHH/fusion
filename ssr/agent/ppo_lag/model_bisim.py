
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
import torch.nn.functional as F

from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.torch_utils import convert_to_torch_tensor, explained_variance, sequence_mask
from transition_dynamics.transition_model import ProbabilisticTransitionModel
from transition_dynamics.bisim_model_head import BisimEncoder_Head, BisimEncoder_Head_BP


def reparameterize(mu, logstd):
    std = torch.exp(logstd)
    eps = torch.randn_like(std)
    return mu + eps * std

class TorchCustomModel(TorchModelV2, nn.Module):
    """Example of a PyTorch custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.hidden_dim = 64
        self.output_dim = 35
        self.lidar_dim = 240
        
        self.bisim_model = BisimEncoder_Head_BP(hidden_dim=self.hidden_dim, output_dim=self.output_dim, causal=True)
        self.bisim_model.load_state_dict(torch.load('/home/haohong/0_causal_drive/baselines_clean/transition_dynamics/bisim_head_state_est_bpr.pt', map_location="cpu"))
        print('bisim model loaded.')
        
        self.fc_act = nn.Sequential(*[
            # nn.Linear(self.output_dim*2+self.lidar_dim+self.output_dim, self.hidden_dim*4),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 4)])
        
        self.fc_val = nn.Sequential(*[
            # nn.Linear(self.output_dim+self.hidden_dim+self.output_dim, self.hidden_dim),
            nn.Linear(self.output_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1)])

        # self.enc_lidar = nn.Sequential(*[
        #     nn.Linear(240, self.hidden_dim), 
        #     nn.ReLU(), 
        #     nn.Linear(self.hidden_dim, self.hidden_dim), 
        #     nn.ReLU()
        # ])

        self._feat = None
        
        # p = list(self.bisim_model.state_encoder_reward.parameters()) + \
        #     list(self.bisim_model.state_encoder_cost.parameters()) + \
        #     list(self.bisim_model.action_encoder.parameters())
        p = list(self.bisim_model.parameters())
        for v in p:
            v.requires_grad = False
        self.criterion = nn.MSELoss()

    def forward(self, input_dict, state, seq_lens):
        inputs = input_dict["obs"]['img'].float()     
        # inputs_lidar = input_dict["obs"]["lidar"].float()
        # z_lidar = self.enc_lidar(inputs_lidar)
        ############# Safe Env ##############
        # inputs_state = input_dict["obs"]["state"].float()
        # logits_out = self.fc_act(inputs_state)
        # self.cur_val = self.fc_val(inputs_state)
        # self._feat = inputs_state
        ############# Safe Env ##############
        
        # inputs = input_dict["obs"]  
        encoder_feat_1 = self.bisim_model.get_z_reward(inputs)
        encoder_feat_cost = self.bisim_model.get_z_cost(inputs)
        self._feat = encoder_feat_cost
        encoder_act = self.bisim_model.action_encoder(inputs)[0]
        logits_out = self.fc_act(encoder_act)
        
        # logits_out = self.bisim_model.get_z_action(inputs)
        self.cur_val = self.fc_val(encoder_feat_1)
        
        # self.cur_val = self.fc_val(z_cat)
        # self._feat = torch.cat([encoder_feat_cost, z_lidar, inputs_state], dim=-1)

        # z_cat = torch.cat([encoder_feat_1, encoder_feat_cost, z_lidar, inputs_state], dim=-1)
        # logits_out = self.fc_act(z_cat)
        # print('=============FUSED FORWARD=============')
        # logits_out = self.bisim_model.get_z_action(inputs)

        ## BC Debug
        # act_expert = input_dict["obs"]["expert"].float()
        # act_expert = torch.clamp(act_expert, -torch.ones_like(act_expert), torch.ones_like(act_expert))
        # act_pred = reparameterize(logits_out[:, :2], logits_out[:, 2:])
        # self.loss_bc = self.criterion(act_pred, act_expert)

        return logits_out, state

    def value_function(self):
        return torch.reshape(self.cur_val, [-1])
    
# class CostValueNetwork(TorchModelV2):
class CostValueNetwork(TorchCustomModel):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(CostValueNetwork, self).__init__(obs_space, action_space, num_outputs, model_config, name)   
        self.hidden_dim = 64
        self.fc_cost_val = nn.Sequential(*[
            # nn.Linear(self.output_dim+self.hidden_dim+self.output_dim, self.hidden_dim),
            nn.Linear(self.output_dim, self.hidden_dim),
            # nn.Linear(275, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1)
        ])
        
        self._last_cost_value = None
        # print('cost network: ', self)
        
    def forward(self, input_dict, state, seq_lens):
        # print('true state in: ', input_dict.keys())
        # input()
        ret = super(CostValueNetwork, self).forward(input_dict, state, seq_lens)
        self._last_cost_value = self.fc_cost_val(self._feat)
        # self._last_cost_value = self.fc_cost_val(input_dict['obs'])
        
        # print('=============FUSED FORWARD COST=============')        
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