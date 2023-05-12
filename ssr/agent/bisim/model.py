
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

from transition_dynamics.encoder import ImageStateEncoder, ImageStateEncoder_NonCausal
from transition_dynamics.transition_model import ProbabilisticTransitionModel

class TorchCustomModel(TorchModelV2, nn.Module):
    """Example of a PyTorch custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        

        # self.torch_sub_model = BetaVAE_H(z_dim=32, nc=3)
        # self.torch_sub_model.load_state_dict(torch.load('/home/haohong/0_MTSRL/baselines_clean/VAE/ckpt/ckpt_archive/32_latent_1000000', map_location="cpu")['model_states']['net'])
        
        self.torch_sub_model = ImageStateEncoder()
        self.torch_sub_model.load_state_dict(torch.load('/home/haohong/0_causal_drive/baselines_clean/transition_dynamics/encoder_causal.pt', map_location='cpu'))
        # self.torch_sub_model = ImageStateEncoder_NonCausal()
        # self.torch_sub_model.load_state_dict(torch.load('/home/haohong/0_causal_drive/baselines_clean/transition_dynamics/encoder_noncausal.pt', map_location='cpu'))
        self.hidden_dim = 64

        
        self.dynamics_model = ProbabilisticTransitionModel(input_dim=19, encoder_feature_dim=self.hidden_dim, \
                        action_shape=(2,), layer_width=self.hidden_dim)
        self.fc_act = nn.Sequential(*[
            nn.Linear(19, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 4)])

        self.fc_val = nn.Sequential(*[
            nn.Linear(19, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1)])
        
        self._feat = None
        
        # p = list(self.torch_sub_model.parameters())        
        # for v in p:
        #     v.requires_grad = False
        
    def forward(self, input_dict, state, seq_lens):
        try:
            inputs = input_dict['obs']['img'].float()     
            _, mu, _ = self.torch_sub_model(inputs)
            self._feat = mu
            logits_out = self.fc_act(mu)
            self.cur_val = self.fc_val(mu)
        except: 
            print('exception in act network')
            logits_out = self.fc_act(self._feat)
        
        return logits_out, state


    def get_bisim_loss(self, input_dict):
        return torch.tensor(0.), torch.tensor(0.) # debug no bisimulation
        try:
            state = restore_original_dimensions(input_dict['obs'], \
            self.obs_space, 'torch') 
            state = state['img']
            input_action = input_dict['actions']
            reward = input_dict['rewards']
            state_next = restore_original_dimensions(input_dict['new_obs'], \
                self.obs_space, 'torch') 
            state_next = self.torch_sub_model(state_next['img'])[0].detach()
                
            # print(state.shape, input_action.shape)
        except:
            print('warming up sample batch...')
            return torch.tensor(0.), torch.tensor(0.)
        encoder_feat_1 = self.dynamics_model.input_encoder(self.torch_sub_model(state)[0])

        batch_size = state.shape[0]
        if batch_size > 1:
            perm_idx = np.random.permutation(batch_size)
            encoder_feat_2 = encoder_feat_1[perm_idx]
            reward_2 = reward[perm_idx]

            # with torch.no_grad():
            pred_mu_1, pred_sigma_1 = self.dynamics_model.predict(encoder_feat_1, input_action)
            pred_mu_2, pred_sigma_2 = pred_mu_1.detach()[perm_idx], pred_sigma_1.detach()[perm_idx]
             
            z_dist = F.smooth_l1_loss(encoder_feat_1, encoder_feat_2, reduce='none')
            r_dist = F.smooth_l1_loss(reward, reward_2, reduce='none')
            transition_dist = torch.sqrt((pred_mu_1.detach()-pred_mu_2).pow(2) + (
                pred_sigma_1.detach()-pred_sigma_2).pow(2))
            bisimilarity = r_dist + 0.99 * transition_dist
            loss_bisim = (z_dist - bisimilarity).pow(2).mean()
        else:
            loss_bisim = torch.zeros(1,)


        diff_dynamics = (pred_mu_1 - state_next) / pred_sigma_1
        loss_dynamics = torch.mean(0.5 * diff_dynamics.pow(2) + torch.log(pred_sigma_1))
        
        # print(loss_bisim, loss_dynamics)
        return loss_bisim, loss_dynamics
            

    def value_function(self):
        return torch.reshape(self.cur_val, [-1])

    
from ray.rllib.models.modelv2 import restore_original_dimensions
class CostValueNetwork(TorchCustomModel):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(CostValueNetwork, self).__init__(obs_space, action_space, num_outputs, model_config, name)   
        
        self.fc_cost_val = nn.Sequential(*[
            nn.Linear(19, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1)
        ])
        
        self._last_cost_value = None
        # print('cost network: ', self)
        
    def forward(self, input_dict, state, seq_lens):
        # print('true state in: ', input_dict.keys())
        # try:
        #     new_obs = restore_original_dimensions(input_dict['new_obs'], self.obs_space, 'torch')        
        # except: 
        #     print('lollllllll')
        
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
        "model_img", TorchCustomModel
    )