"""
this extremely minimal Decision Transformer model is based on
the following causal transformer (GPT) implementation:

Misha Laskin's tweet:
https://twitter.com/MishaLaskin/status/1481767788775628801?cxt=HHwWgoCzmYD9pZApAAAA

and its corresponding notebook:
https://colab.research.google.com/drive/1NUBqyboDcGte5qAJKOl8gaJC28V_73Iv?usp=sharing

** the above colab notebook has a bug while applying masked_fill 
which is fixed in the following code
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.networks import build_encoder_net
from utils.utils import CUDA, CPU

def reparameterize(mu, std):
    # std = torch.exp(logstd)
    eps = torch.randn_like(std)
    return mu + eps * std

class Encoder(nn.Module):
    def __init__(self, hidden_dim, nc, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nc = nc
        
        self.encoder = build_encoder_net(hidden_dim*nc, nc)
        self.fc_out = nn.Linear(hidden_dim*nc, output_dim*2)
        # self.ln = nn.LayerNorm(OBS_DIM)

    def forward(self, x): 

        z_tmp = self.encoder(x)
        mu, std = z_tmp[:, :self.hidden_dim*self.nc], z_tmp[:, self.hidden_dim*self.nc:]
        z_tmp = reparameterize(mu, std)
        
        output = self.fc_out(z_tmp)

        mu, std = output[:, :self.output_dim], output[:, self.output_dim:]
        state_pred = reparameterize(mu, std)

        return state_pred, mu, torch.exp(std)
    
class ImageEncoder(nn.Module):
    def __init__(self, hidden_dim, output_dim) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lidar_dim = 240
        self.action_encoder = Encoder(hidden_dim=hidden_dim, nc=5, output_dim=output_dim)
        
        self.criteria = nn.MSELoss(reduction='mean')        
        
        # self.opt_actor = optim.Adam(self.parameters(), lr=1e-4, betas=(0.9, 0.999))
        # self.opt_critic = optim.Adam(self.parameters(), lr=1e-4, betas=(0.9, 0.999))
        self.sigma_min = 1e-2
        self.sigma_max = 1.
    
    def forward(self, img, state, deterministic=False):
        '''
            Get the cost-aware Bisimulation Loss for the representation
        '''
        if not deterministic: 
            state_pred, _, _ = self.action_encoder(img)
        else:
            _, state_pred, _ = self.action_encoder(img)

        loss_state = self.criteria(state_pred, state)
        
        return state_pred, loss_state
    
    @torch.no_grad()
    def encode(self, img): 
        return self.action_encoder(img)[1]
    
class MaskedCausalAttention(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()

        self.n_heads = n_heads
        self.max_T = max_T

        self.q_net = nn.Linear(h_dim, h_dim)
        self.k_net = nn.Linear(h_dim, h_dim)
        self.v_net = nn.Linear(h_dim, h_dim)

        self.proj_net = nn.Linear(h_dim, h_dim)

        self.att_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)

        ones = torch.ones((max_T, max_T))
        mask = torch.tril(ones).view(1, 1, max_T, max_T)

        # register buffer makes sure mask does not get updated
        # during backpropagation
        self.register_buffer('mask',mask)

    def forward(self, x):
        B, T, C = x.shape # batch size, seq length, h_dim * n_heads

        N, D = self.n_heads, C // self.n_heads # N = num heads, D = attention dim

        # rearrange q, k, v as (B, N, T, D)
        q = self.q_net(x).view(B, T, N, D).transpose(1,2)
        k = self.k_net(x).view(B, T, N, D).transpose(1,2)
        v = self.v_net(x).view(B, T, N, D).transpose(1,2)

        # weights (B, N, T, T)
        weights = q @ k.transpose(2,3) / math.sqrt(D)
        # causal mask applied to weights
        weights = weights.masked_fill(self.mask[...,:T,:T] == 0, float('-inf'))
        # normalize weights, all -inf -> 0 after softmax
        normalized_weights = F.softmax(weights, dim=-1)

        # attention (B, N, T, D)
        attention = self.att_drop(normalized_weights @ v)

        # gather heads and project (B, N, T, D) -> (B, T, N*D)
        attention = attention.transpose(1, 2).contiguous().view(B,T,N*D)

        out = self.proj_drop(self.proj_net(attention))
        return out, normalized_weights


class Block(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()
        self.attention = MaskedCausalAttention(h_dim, max_T, n_heads, drop_p)
        self.mlp = nn.Sequential(
                nn.Linear(h_dim, 4*h_dim),
                nn.GELU(),
                nn.Linear(4*h_dim, h_dim),
                nn.Dropout(drop_p),
            )
        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)
        self.count = 0
    def forward(self, x):
        # Attention -> LayerNorm -> MLP -> LayerNorm
        x = x + self.attention(x)[0] # residual
        x = self.ln1(x)
        x = x + self.mlp(x) # residual
        x = self.ln2(x)
        return x


class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_dim, n_blocks, h_dim, context_len,
                 n_heads, drop_p, max_timestep=4096):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.h_dim = h_dim

        ### transformer blocks
        input_seq_len = 3 * context_len
        blocks = [Block(h_dim, input_seq_len, n_heads, drop_p) for _ in range(n_blocks)]
        self.transformer = nn.Sequential(*blocks)
        
        ### projection heads (project to embedding)
        self.embed_ln = nn.LayerNorm(h_dim)
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        self.embed_rtg = torch.nn.Linear(1, h_dim)
        self.embed_state = torch.nn.Linear(state_dim, h_dim)

        # # discrete actions
        # self.embed_action = torch.nn.Embedding(act_dim, h_dim)
        # use_action_tanh = False # False for discrete actions

        # continuous actions
        self.embed_action = torch.nn.Linear(act_dim, h_dim)
        use_action_tanh = True # True for continuous actions

        ### prediction heads
        self.predict_rtg = torch.nn.Linear(h_dim, 1)
        self.predict_state = torch.nn.Linear(h_dim, state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(h_dim, act_dim)] + ([nn.Tanh()] if use_action_tanh else []))
        )


    def forward(self, timesteps, states, actions, returns_to_go):

        B, T, _ = states.shape

        time_embeddings = self.embed_timestep(timesteps)
        # print(self.embed_action.parameters(), actions)
        # time embeddings are treated similar to positional embeddings
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        returns_embeddings = self.embed_rtg(returns_to_go) + time_embeddings

        # stack rtg, states and actions and reshape sequence as
        # (r_0, s_0, a_0, r_1, s_1, a_1, r_2, s_2, a_2 ...)
        h = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(B, 3 * T, self.h_dim)

        h = self.embed_ln(h)

        # transformer and prediction
        h = self.transformer(h)

        # get h reshaped such that its size = (B x 3 x T x h_dim) and
        # h[:, 0, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t
        # h[:, 1, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t, s_t
        # h[:, 2, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t, s_t, a_t
        # that is, for each timestep (t) we have 3 output embeddings from the transformer,
        # each conditioned on all previous timesteps plus 
        # the 3 input variables at that timestep (r_t, s_t, a_t) in sequence.
        h = h.reshape(B, T, 3, self.h_dim).permute(0, 2, 1, 3)

        # get predictions
        return_preds = self.predict_rtg(h[:,2])     # predict next rtg given r, s, a
        state_preds = self.predict_state(h[:,2])    # predict next state given r, s, a
        action_preds = self.predict_action(h[:,1])  # predict action given r, s

        return state_preds, action_preds, return_preds



class Fusion(nn.Module):
    def __init__(self, state_dim, act_dim, n_blocks, h_dim, context_len,
                 n_heads, drop_p, max_timestep=4096):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.h_dim = h_dim

        ### transformer blocks
        input_seq_len = 4 * context_len
        blocks = [Block(h_dim, input_seq_len, n_heads, drop_p) for _ in range(n_blocks)]
        self.transformer = nn.Sequential(*blocks)

        ### projection heads (project to embedding)
        self.embed_ln = nn.LayerNorm(h_dim)
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        self.embed_rtg = torch.nn.Linear(1, h_dim)
        self.embed_ctg = torch.nn.Linear(1, h_dim)
                
        self.embed_state = torch.nn.Linear(state_dim, h_dim)

        # # discrete actions
        # self.embed_action = torch.nn.Embedding(act_dim, h_dim)
        # use_action_tanh = False # False for discrete actions

        # continuous actions
        self.embed_action = torch.nn.Linear(act_dim, h_dim)
        use_action_tanh = True # True for continuous actions

        ### prediction heads
        self.predict_rtg = torch.nn.Linear(h_dim, 1)
        self.predict_ctg = torch.nn.Linear(h_dim, 1)
        
        self.predict_state = torch.nn.Linear(h_dim, state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(h_dim, act_dim)] + ([nn.Tanh()] if use_action_tanh else []))
        )


    def forward(self, timesteps, states, actions, returns_to_go, returns_to_go_cost):

        B, T, _ = states.shape

        time_embeddings = self.embed_timestep(timesteps)
        # print(self.embed_action.parameters(), actions)
        # time embeddings are treated similar to positional embeddings
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        returns_embeddings = self.embed_rtg(returns_to_go) + time_embeddings
        returns_embeddings_cost = self.embed_ctg(returns_to_go_cost) + time_embeddings
        
        # stack rtg, states and actions and reshape sequence as
        # (r_0, s_0, a_0, r_1, s_1, a_1, r_2, s_2, a_2 ...)
        h = torch.stack(
            (returns_embeddings_cost, returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(B, 4 * T, self.h_dim)

        h = self.embed_ln(h)

        # transformer and prediction
        h = self.transformer(h)

        # get h reshaped such that its size = (B x 3 x T x h_dim) and
        # h[:, 0, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t
        # h[:, 1, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t, s_t
        # h[:, 2, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t, s_t, a_t
        # that is, for each timestep (t) we have 3 output embeddings from the transformer,
        # each conditioned on all previous timesteps plus 
        # the 3 input variables at that timestep (r_t, s_t, a_t) in sequence.
        h = h.reshape(B, T, 4, self.h_dim).permute(0, 2, 1, 3)

        # get predictions
        return_cost_preds = self.predict_ctg(h[:,3])     # predict next rtg given c, r, s, a
        return_preds = self.predict_rtg(h[:,3])     # predict next rtg given c, r, s, a
        state_preds = self.predict_state(h[:,3])    # predict next state given c, r, s, a
        action_preds = self.predict_action(h[:,2])  # predict action given c, r, s

        return state_preds, action_preds, return_preds, return_cost_preds


class Fusion_Structure(nn.Module):
    def __init__(self, state_dim, act_dim, n_blocks, h_dim, context_len,
                 n_heads, drop_p, h_dim_pretrained=64, max_timestep=4096):
        super().__init__()

        self.state_dim = state_dim
        self.lidar_dim = 240
        self.act_dim = act_dim
        self.h_dim = h_dim

        ### transformer blocks
        input_seq_len = 4 * context_len
        blocks = [Block(h_dim, input_seq_len, n_heads, drop_p) for _ in range(n_blocks)]
        self.transformer = nn.Sequential(*blocks)
        
        ### projection heads (project to embedding)
        self.embed_ln = nn.LayerNorm(h_dim)
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        self.embed_rtg = torch.nn.Linear(1, h_dim)
        self.embed_ctg = torch.nn.Linear(1, h_dim)
        self.embed_lidar = nn.Linear(self.lidar_dim, h_dim)
        self.embed_agg = nn.Sequential(*[nn.Linear(h_dim*3, h_dim),
                                        nn.ReLU(),
                                        nn.Linear(h_dim, h_dim)])
        
        self.embed_img = CUDA(ImageEncoder(hidden_dim=h_dim_pretrained, output_dim=state_dim))
        self.embed_img.load_state_dict(torch.load('checkpoint/state_est.pt'))
        for v in self.embed_img.parameters():
            v.requires_grad = False
            
        self.embed_state = torch.nn.Linear(state_dim, h_dim)
        
        # continuous actions
        self.embed_action = torch.nn.Linear(act_dim, h_dim)
        use_action_tanh = True # True for continuous actions

        ### prediction heads
        self.predict_rtg = torch.nn.Linear(h_dim, 1)
        self.predict_ctg = torch.nn.Linear(h_dim, 1)
        
        self.predict_state = torch.nn.Linear(h_dim, state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(h_dim, act_dim)] + ([nn.Tanh()] if use_action_tanh else []))
        )


    def forward(self, timesteps, states, actions, returns_to_go, returns_to_go_cost, deterministic=False):
        state, lidar, img = states  
        B, T, _ = state.shape
        if state.isnan().any(): 
            state = torch.nan_to_num(state, nan=0.0)
            print('replace nan values.')
        
        time_embeddings = self.embed_timestep(timesteps)
        # # time embeddings are treated similar to positional embeddings
        ego_embeddings = self.embed_state(state)        
        lidar_embeddings = self.embed_lidar(lidar)      
        # bev embedding will be replaced by the image encoder at the inference time  
        if deterministic: 
            img = img.reshape(-1, 5, 84, 84)
            bev_embeddings = self.embed_img.encode(img)
            bev_embeddings = bev_embeddings.reshape(B, T, -1)
            bev_embeddings = self.embed_state(bev_embeddings)
        else: 
            bev_embeddings = ego_embeddings.clone()
            
        state_embeddings = self.embed_agg(torch.cat([ego_embeddings, lidar_embeddings, bev_embeddings], dim=2)) + time_embeddings
        
        action_embeddings = self.embed_action(actions) + time_embeddings
        returns_embeddings = self.embed_rtg(returns_to_go) + time_embeddings
        returns_embeddings_cost = self.embed_ctg(returns_to_go_cost) + time_embeddings
         
        # stack rtg, states and actions and reshape sequence as
        # (c_0, r_0, s_0, a_0, c_1, r_1, s_1, a_1, c_2, r_2, s_2, a_2 ...)
        h = torch.stack(
            (returns_embeddings_cost, returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(B, 4 * T, self.h_dim)
        
        # print(returns_to_go_cost.shape)
        torch.save(returns_to_go_cost, 'cost.pt')
        h = self.embed_ln(h)
        # print(h.shape)
        # transformer and prediction
        h = self.transformer(h)
        
        # get h reshaped such that its size = (B x 4 x T x h_dim)
        # that is, for each timestep (t) we have 3 output embeddings from the transformer,
        # each conditioned on all previous timesteps plus 
        h = h.reshape(B, T, 4, self.h_dim).permute(0, 2, 1, 3)

        # get predictions conditioned on the attended "causal parents"
        return_cost_preds = self.predict_ctg(h[:,3])     # predict next rtg given c, r, s, a
        return_preds = self.predict_rtg(h[:,3])     # predict next rtg given c, r, s, a
        state_preds = self.predict_state(h[:,3])    # predict next state given c, r, s, a
        action_preds = self.predict_action(h[:,2])  # predict action given c, r, s

        return state_preds, action_preds, return_preds, return_cost_preds


if __name__ == '__main__': 
    #### Toy Example
    model = Fusion_Structure(state_dim=35, act_dim=2, n_blocks=3, h_dim=64, context_len=30, n_heads=8, drop_p=0.1, max_timestep=1000)
    timestep = torch.randint(0, 1, (128, 1))
    rtg = torch.rand(128, 30, 1)
    state = torch.rand(128, 30, 35)
    act = torch.rand(128, 30, 2)
    
    data = torch.rand(128, 64)
    s, a, r = model.forward(timestep, state, act, rtg)
    print(s.shape, a.shape, r.shape)
    pass
    