
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

from ssr.transition_dynamics.transition_model import ProbabilisticTransitionModel

from ssr.encoder.model_actor import BisimEncoder_Head_BP_Actor
from ssr.encoder.model_critic import BisimEncoder_Head_BP_Value_Net
# from ssr.encoder.model_critic_cost import BisimEncoder_Head_BP_Cost






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
        
        self.actor_model = BisimEncoder_Head_BP_Actor(hidden_dim=self.hidden_dim, output_dim=self.output_dim, causal=True)
        # self.actor_model.load_state_dict(torch.load('/home/haohong/0_causal_drive/baselines_clean/transition_dynamics/attention_encoder_actor_large_var.pt', map_location="cpu"))
        
        self.value_model = BisimEncoder_Head_BP_Value_Net(hidden_dim=self.hidden_dim, output_dim=self.output_dim, causal=True)
        # self.value_model.load_state_dict(torch.load('/home/haohong/0_causal_drive/ssr-rl/checkpoint/joint_critic.pt', map_location="cpu"))

        # self.value_cost_model = BisimEncoder_Head_BP_Cost(hidden_dim=self.hidden_dim, output_dim=self.output_dim, causal=True)
        # self.value_cost_model.load_state_dict(torch.load('/home/haohong/0_causal_drive/ssr-rl/checkpoint/attention_encoder_value_cost.pt', map_location="cpu"))

        print('bisim model loaded.')
        
        self._feat = None
        # self.fc_val = nn.Sequential(*[
        #     nn.Linear(self.hidden_dim, self.hidden_dim),
        #     nn.ReLU(), 
        #     nn.Linear(self.hidden_dim, 1)  
        # ])
        # p = list(self.bisim_model.state_encoder_reward.parameters()) + \
        #     list(self.bisim_model.state_encoder_cost.parameters()) + \
        #     list(self.bisim_model.action_encoder.parameters())
        p = list(self.actor_model.parameters()) + list(self.value_model.parameters())
        for v in p:
            v.requires_grad = False
        
        p = list(self.actor_model.action_encoder.parameters()) + list(self.actor_model.action_estimator.parameters()) \
            + list(self.value_model.value_encoder.parameters()) + list(self.value_model.value_estimator.parameters()) + \
                list(self.value_model.lidar_encoder.parameters()) + list(self.value_model.state_encoder.parameters())
        for v in p:
            v.requires_grad = True
        
        print('gradient setup')
        
        self.criterion = nn.MSELoss()
        self.sigma_max = 1.
        self.sigma_min = 1e-2
        
    def forward(self, input_dict, state, seq_lens):
        inputs = input_dict["obs"]['img'].float()     
        lidar = input_dict["obs"]["lidar"].float()
        input_state = input_dict["obs"]["state"].float()
        
        
        z_act, _, _ = self.actor_model.action_encoder(inputs)
        act_pred = self.actor_model.action_estimator(z_act)
        mu_act, std_act = act_pred[:, :2], act_pred[:, 2:]
        std_act = torch.sigmoid(std_act)
        std_act = torch.log(self.sigma_min + (self.sigma_max-self.sigma_min)*std_act)
        # std_act = torch.clamp(std_act, np.log(self.sigma_min), np.log(self.sigma_max))
        logits_out = torch.cat([mu_act, std_act], dim=-1)
        # print(logits_out)
        
        z_img = self.value_model.value_encoder(inputs)
        z_lidar = self.value_model.lidar_encoder(lidar)
        z_ego = self.value_model.state_encoder(input_state)
        # print(z_img.shape, z_lidar.shape, z_ego.shape)
            
        z_reward = torch.cat([z_img[:, :self.hidden_dim*2], z_ego], dim=-1)
        z_cost = torch.cat([z_img[:, self.hidden_dim*2:], z_lidar, z_ego], dim=-1)
        z_reward = self.value_model.agg_reward(z_reward)
        z_cost = self.value_model.agg_cost(z_cost)
        
        self.cur_val = self.value_model.value_estimator(z_reward)*40.+40.

        self.cur_val_cost = self.value_model.value_estimator(z_cost)*10.
        
        
        ##############
        # z_road, _, _ = self.bisim_model.state_encoder_road(inputs)
        # z_vehicles, _, _ = self.bisim_model.state_encoder_vehicles(inputs)
        # z_lidar = self.bisim_model.state_encoder_lidar(lidar)
        
        # z_act = torch.cat([z_road, z_vehicles, z_lidar], dim=-1)
        # z_reward = torch.cat([z_road, z_vehicles, z_lidar], dim=-1)
        # z_cost = torch.cat([z_vehicles, z_lidar], dim=-1)
        # logits_out = self.bisim_model.action_estimator(z_act)
        # self.cur_val = self.bisim_model.fc_vr(z_reward) *40. + 40.
        # self.cur_val_cost = self.bisim_model.fc_vc(z_cost) * 10.
        ###########
        ###############
        # z_act, _, _ = self.bisim_model.action_encoder(inputs)
        # logits_out = self.bisim_model.action_estimator(z_act)
        # self.cur_val = self.fc_val(z_act)
        # self._feat = z_act
        ####################
        
        return logits_out, state

    def value_function(self):
        return torch.reshape(self.cur_val, [-1])
    
# class CostValueNetwork(TorchModelV2):
class CostValueNetwork(TorchCustomModel):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(CostValueNetwork, self).__init__(obs_space, action_space, num_outputs, model_config, name)   
        self.hidden_dim = 64
        # self.fc_cost_val = nn.Sequential(*[
        #     # nn.Linear(self.output_dim+self.hidden_dim+self.output_dim, self.hidden_dim),
        #     # nn.Linear(self.output_dim, self.hidden_dim),
        #     # nn.Linear(275, self.hidden_dim),
        #     # nn.Tanh(),
        #     nn.Linear(self.hidden_dim, self.hidden_dim),
        #     # nn.Tanh(),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_dim, 1)
        # ])
        
        self._last_cost_value = None
        # print('cost network: ', self)
        
    def forward(self, input_dict, state, seq_lens):
        # print('true state in: ', input_dict.keys())
        # input()
        ret = super(CostValueNetwork, self).forward(input_dict, state, seq_lens)
        # self._last_cost_value = self.fc_cost_val(self._feat)
        self._last_cost_value = self.cur_val_cost
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