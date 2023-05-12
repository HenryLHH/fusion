
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

class TorchCustomModel(TorchModelV2, nn.Module):
    """Example of a PyTorch custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.hidden_dim = 16
        self.torch_sub_model = ProbabilisticTransitionModel(input_dim=obs_space.shape[0], encoder_feature_dim=self.hidden_dim, \
                        action_shape=action_space.shape, layer_width=self.hidden_dim)
        self.torch_sub_model.load_state_dict(torch.load('transition_dynamics/transition_model.pt', map_location="cpu"))

        
        self.fc_act = nn.Sequential(*[
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 4)])

        self.fc_val = nn.Sequential(*[
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1)])
        
        self._feat = None
        
        # p = list(self.torch_sub_model.fc_mu.parameters()) + list(self.torch_sub_model.fc_sigma.parameters()) \
        #     + list(self.torch_sub_model.fc.parameters()) 
        # for v in p:
        #     v.requires_grad = False
        
    def forward(self, input_dict, state, seq_lens):
        inputs = input_dict["obs"]['img'].float()     
        # inputs = input_dict["obs"]  
        encoder_feat_1 = self.torch_sub_model.input_encoder(inputs)        
        self._feat = encoder_feat_1
        logits_out = self.fc_act(encoder_feat_1)
        self.cur_val = self.fc_val(encoder_feat_1)
        return logits_out, state

    def value_function(self):
        return torch.reshape(self.cur_val, [-1])
    

class CostValueNetwork(TorchCustomModel):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(CostValueNetwork, self).__init__(obs_space, action_space, num_outputs, model_config, name)   
        
        self.fc_cost_val = nn.Sequential(*[
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1)
        ])
        
        self._last_cost_value = None
        # print('cost network: ', self)
        
    def forward(self, input_dict, state, seq_lens):
        print('true state in: ', input_dict.keys())
        # input()
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