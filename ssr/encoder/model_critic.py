import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from ssr.transition_dynamics.transition_model import ProbabilisticTransitionModel

from utils.utils import CUDA, CPU, reparameterize, kl_divergence
from utils.networks import *



torch.autograd.set_detect_anomaly(True)

OBS_DIM = 35
NUM_FILES = 400000


def eval_loss_dataloader_state(model, dataloader, verbose=0):
    model.eval()
    loss = 0
    accuracy = 0.0
    recall_all, precision_all = 0.0, 0.0

    pbar = tqdm(total=len(dataloader))
    for data in dataloader:
        pbar.update(1)
        img, _, lidar, _, state, state_next, action, reward, cost, vr_target, vc_target = data
        img = CUDA(img)
        lidar = CUDA(lidar)
        
        state = CUDA(state)
        state_next = CUDA(state_next)
        action = CUDA(action)
        reward = CUDA(reward)
        cost = CUDA(torch.LongTensor(cost.reshape(-1)))
        vr_target = (CUDA(vr_target)-40) / 40.
        vc_target = CUDA(vc_target) / 10.
        
        # loss_bisim, loss_bisim_cost = 0., 0.
        if args.causal: 
            loss_est, loss_act, loss_vr,  loss_vc, loss_bisim, loss_bisim_cost = \
                            model(img, lidar, state, action, reward, cost, vr_target, vc_target, 
                                  deterministic=True)
        else: 
            loss_est, loss_ret, pred_cls = model(img, state, action, reward, cost)
            

        loss += (loss_vr + loss_vc).item() # (loss_est+loss_ret+loss_bisim+loss_bisim_cost).item()
        # print('train loss: {:4f} | precision: {:.4f} | recall: {:.4f} | acc: {:.4f} '.format(loss_ret.item(), precision, recall, acc), end='\r')
        torch.cuda.empty_cache()
    
    loss /= len(dataloader)
    recall_all /= len(dataloader)
    precision_all /= len(dataloader)
    accuracy /= len(dataloader)
    print('train loss: {:4f} | state est. loss: {:.4f} | act loss: {:.4f} | V_r loss: {:.4f} | V_c loss: {:.4f} | bisim loss: {:.4f} | bisim loss cost: {:.4f} '.format(
        loss, loss_est.item(), loss_act.item(), loss_vr.item(), loss_vc.item(), loss_bisim.item(), loss_bisim_cost.item()))

    pbar.close()

    return loss

def visualize_ckpt(model, ckpt, dataloader):
    model.load_state_dict(torch.load(ckpt))
    for data in dataloader:
        img, _, state, state_next, action, reward, cost = data
        img = CUDA(img)
        state = CUDA(state)
        state_next = CUDA(state_next)
        action = CUDA(action)
        reward = CUDA(reward)
        cost = CUDA(cost)
        _, _, _ = model(img, state, action, reward, cost)
        print(z.shape)
        cost = cost.detach().cpu().numpy()
        reward = reward.detach().cpu().numpy()

        idx0 = np.where(cost == 0)[0]
        idx1 = np.where(cost != 0)[0]

        z = z.detach().cpu().numpy()
        z_plot = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(z)

        # plt.scatter(z_plot[:, 0], z_plot[:, 1], c=reward)
        plt.scatter(z_plot[idx0, 0], z_plot[idx0, 1])
        plt.scatter(z_plot[idx1, 0], z_plot[idx1, 1])
        plt.legend(['Safe', 'Unsafe'])
        plt.savefig(ckpt.split('.')[0])
        break


def visualize_ckpt_state(model, ckpt, dataloader):
    model.load_state_dict(torch.load(ckpt))
    for data in dataloader:
        img, _, state, state_next, action, reward, cost = data
        img = CUDA(img)
        state = CUDA(state)
        state_next = CUDA(state_next)
        action = CUDA(action)
        reward = CUDA(reward)
        cost = CUDA(torch.LongTensor(cost.reshape(-1)))

        z, loss_ret, pred_cls = model(img, state, action, reward, cost)
        acc = len(torch.where(pred_cls==cost)[0]) / len(cost)
        TP = len(torch.where(torch.logical_and(pred_cls==cost, cost))[0])
        recall = TP / ((len(torch.where(cost)[0]))+1e-8)
        precision = TP / (len(torch.where(pred_cls)[0])+1e-8)

        cost = cost.detach().cpu().numpy()
        reward = reward.detach().cpu().numpy()

        idx0 = np.where(cost == 0)[0]
        idx1 = np.where(cost != 0)[0]

        z = torch.relu(z).detach().cpu().numpy()
        z_plot = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(z)

        # plt.scatter(z_plot[:, 0], z_plot[:, 1], c=reward)
        plt.scatter(z_plot[idx0, 0], z_plot[idx0, 1])
        plt.scatter(z_plot[idx1, 0], z_plot[idx1, 1])
        plt.legend(['Safe', 'Unsafe'])
        plt.savefig(ckpt.split('.')[0])
        break



class BisimEncoder_Head_BP_GRU(nn.Module):
    def __init__(self, hidden_dim, output_dim, causal=False) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lidar_dim = 240
        self.causal = causal
        # self.state_encoder = StateEncoder(obs_dim=35, hidden_dim=hidden_dim)
        if causal:
            self.state_encoder_road = ImageStateEncoder_GRU(hidden_dim=hidden_dim, nc=5, output_dim=hidden_dim)
            self.state_encoder_vehicles = ImageStateEncoder_GRU(hidden_dim=hidden_dim, nc=5, output_dim=hidden_dim)
            self.action_encoder = ImageStateEncoder(hidden_dim=hidden_dim, nc=5, output_dim=hidden_dim)

        else:
            self.state_encoder = ImageStateEncoder_NonCausal(hidden_dim=hidden_dim, nc=5, output_dim=hidden_dim)

        self.state_encoder_lidar = nn.Sequential(*[
            nn.Linear(self.lidar_dim, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim)  
        ])
        
        self.state_estimator = nn.Sequential(*[
            nn.Linear(hidden_dim*3, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, output_dim)  
        ])
        
        self.action_estimator = nn.Sequential(*[
            nn.Linear(hidden_dim*3, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, 4)  
        ])

        self.fc_vr = nn.Sequential(*[
            nn.Linear(hidden_dim*3, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, 1)  
        ])
        
        self.fc_vc = nn.Sequential(*[
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, 1)  
        ])
        
        self.state_dynamics = ProbabilisticTransitionModel(input_dim=35, encoder_feature_dim=64, action_shape=(2, ), layer_width=64)
        self.state_dynamics.load_state_dict(torch.load('/home/haohong/0_causal_drive/baselines_clean/transition_dynamics/transition_model_full.pt',\
                                            ))  # map_location="cpu"))
        # self.state_dynamics.load_state_dict(torch.load('transition_model_full.pt'))
        
        self.criterion_est = nn.MSELoss()
        self.criterion_val = nn.MSELoss('mean')
        
        self.sigma_min = 1e-2
        self.sigma_max = 1
        
    def _compute_bisim(self, z, state, action, r):
        batch_size = state.shape[0]
        perm_idx = np.random.permutation(batch_size)

        mu_state_next, sigma_state_next = self.state_dynamics(state, action)
        mu_state_next_2 = mu_state_next.detach()[perm_idx]
        sigma_state_next_2 = sigma_state_next.detach()[perm_idx]

        
        ### task 1: (causal) state estimation

        ### task 2: cost bisimulation metrics
        z_2 = z[perm_idx]
        r_2 = r[perm_idx]
        
        z_dist = F.smooth_l1_loss(z, z_2)
        r_dist = F.smooth_l1_loss(r.float(), r_2.float())
        trans_dist = torch.sqrt((mu_state_next.detach()-mu_state_next_2).pow(2) + (
                                sigma_state_next.detach()-sigma_state_next_2).pow(2))

        bisimilarity = r_dist + 0.99 * trans_dist

        loss_bisim = (z_dist - bisimilarity).pow(2).mean()

        return loss_bisim        
        


    def forward(self, img, lidar, state, action, reward, cost, vr_target, vc_target):
        '''
            Get the cost-aware Bisimulation Loss for the representation
        '''
        if self.causal:
            z_road, _, _ = self.state_encoder_road(img)
            z_vehicles, _, _ = self.state_encoder_vehicles(img)
            z_lidar = self.state_encoder_lidar(lidar)
            
            z_act = torch.cat([z_road, z_vehicles, z_lidar], dim=-1)
            z_reward = torch.cat([z_road, z_vehicles, z_lidar], dim=-1)
            z_cost = torch.cat([z_vehicles, z_lidar], dim=-1)
            
            
            # a_pred = self.action_estimator(z_act)
            v_pred = self.fc_vr(z_reward)
            v_pred_cost = self.fc_vc(z_cost)
            
            # z_cost, _, _ = self.state_encoder_cost(img)
            # z_reward, _, _ = self.state_encoder_reward(img)
            z_act, _, _ = self.action_encoder(img)
            a_pred = self.action_estimator(z_act)
            
            mu_act, std_act = a_pred[:, :2], a_pred[:, 2:]
            std_act = torch.sigmoid(std_act)
            std_act = self.sigma_min + (self.sigma_max-self.sigma_min)*std_act
            act = reparameterize(mu_act, std_act)
            
            loss_act = self.criterion_est(act, action)
            
            # z = self.state_encoder(state)
            pred_state = self.state_estimator(z_reward)
            loss_est = self.criterion_est(pred_state, state)
            # pred_state = self.state_estimator(z_cost)
            # loss_est += self.criterion_est(pred_state, state)

            loss_vr = self.criterion_val(v_pred, vr_target)
            loss_vc = self.criterion_val(v_pred_cost, vc_target)
            
            loss_bisim_r = torch.tensor(0.) # self._compute_bisim(z_reward, state, action, reward)
            loss_bisim_c = torch.tensor(0.) # self._compute_bisim(z_cost, state, action, cost)
            
            return loss_est, loss_act, loss_vr, loss_vc, loss_bisim_r, loss_bisim_c
        
        else:
            z, _, _ = self.state_encoder(img)
            pred = self.state_classifier(z)
            pred_state = self.state_estimator(z_cost, z_reward)
            loss_est = self.criterion_est(pred_state, state)

            loss_cls = self.criterion(pred, cost)
            pred_cls = torch.argmax(pred.detach(), dim=1)

            return loss_est, loss_cls, pred_cls

        # return state_next_est, z, kl_loss, loss_bisim, loss_bisim_cost

    def get_z_reward(self, img): 
        return self.state_estimator(self.state_encoder_reward(img)[1])
        
    def get_z_cost(self, img): 
        return self.state_estimator(self.state_encoder_cost(img)[1])
    
    def get_z_action(self, img): 
        return self.action_estimator(self.action_encoder(img)[0])
    
    def get_z(self, img): 
        return self.state_encoder(img)

class BisimEncoder_Head_BP_Value(nn.Module):
    def __init__(self, hidden_dim, output_dim, causal=False) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lidar_dim = 240
        self.causal = causal
        # self.state_encoder = StateEncoder(obs_dim=35, hidden_dim=hidden_dim)
        if causal:
            # self.state_encoder_road = ImageStateEncoder_GRU(hidden_dim=hidden_dim, nc=5, output_dim=hidden_dim)
            # self.state_encoder_vehicles = ImageStateEncoder_GRU(hidden_dim=hidden_dim, nc=5, output_dim=hidden_dim)
            self.value_encoder = ImageStateEncoder(hidden_dim=hidden_dim, nc=5, output_dim=hidden_dim)
            # self.value_encoder = ImageStateEncoder(hidden_dim=hidden_dim, nc=5, output_dim=hidden_dim)
            # self.value_encoder_cost = ImageStateEncoder(hidden_dim=hidden_dim, nc=5, output_dim=hidden_dim)
            
        else:
            self.state_encoder = ImageStateEncoder_NonCausal(hidden_dim=hidden_dim, nc=5, output_dim=hidden_dim)
        
        self.value_estimator = nn.Sequential(*[
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, 1)  
        ])
        
        self.criterion_est = nn.MSELoss()
        self.criterion_val = nn.MSELoss('mean')

        # self.opt_actor = optim.Adam(self.parameters(), lr=1e-4, betas=(0.9, 0.999))
        # self.opt_critic = optim.Adam(self.parameters(), lr=1e-4, betas=(0.9, 0.999))
        self.sigma_min = 1e-2
        self.sigma_max = 1.
        
        # self.state_dynamics = ProbabilisticTransitionModel(input_dim=35, encoder_feature_dim=64, action_shape=(2, ), layer_width=64)
        # self.state_dynamics.load_state_dict(torch.load('/home/haohong/0_causal_drive/baselines_clean/transition_dynamics/transition_model_full.pt',\
        #                                                 map_location="cpu"
        #                                                 ))        
    def _compute_bisim(self, z, state, action, r):
        batch_size = state.shape[0]
        perm_idx = np.random.permutation(batch_size)

        mu_state_next, sigma_state_next = self.state_dynamics(state, action)
        mu_state_next_2 = mu_state_next.detach()[perm_idx]
        sigma_state_next_2 = sigma_state_next.detach()[perm_idx]

        
        ### task 1: (causal) state estimation

        ### task 2: cost bisimulation metrics
        z_2 = z[perm_idx]
        r_2 = r[perm_idx]
        
        z_dist = F.smooth_l1_loss(z, z_2)
        r_dist = F.smooth_l1_loss(r.float(), r_2.float())
        trans_dist = torch.sqrt((mu_state_next.detach()-mu_state_next_2).pow(2) + (
                                sigma_state_next.detach()-sigma_state_next_2).pow(2))

        bisimilarity = r_dist + 0.99 * trans_dist

        loss_bisim = (z_dist - bisimilarity).pow(2).mean()

        return loss_bisim        


    def forward(self, img, lidar, state, action, reward, cost, vr_target, vc_target, deterministic=False):
        '''
            Get the cost-aware Bisimulation Loss for the representation
        '''
        if self.causal:
            
            if deterministic: 
                _, z_val, _ = self.value_encoder(img)
            else:
                z_val, _, _ = self.value_encoder(img)
            
            pred_vr = self.value_estimator(z_val)
            loss_vr = self.criterion_val(pred_vr, vr_target)
            
            # pred_vc = self.value_estimator(z_act)
            # loss_vc = self.criterion_val(pred_vc, vc_target)
            
            loss_est = torch.tensor(0.)
            loss_act = torch.tensor(0.)
            loss_vc = torch.tensor(0.)

            loss_bisim_r = self._compute_bisim(z_val, state, action, reward)
            loss_bisim_c = torch.tensor(0.) # self._compute_bisim(z_cost, state, action, cost)
            
            return loss_est, loss_act, loss_vr, loss_vc, loss_bisim_r, loss_bisim_c
        
        else:
            raise ValueError('none')

            return loss_est, loss_cls, pred_cls

        # return state_next_est, z, kl_loss, loss_bisim, loss_bisim_cost

    def get_z_reward(self, img): 
        return self.state_estimator(self.state_encoder_reward(img)[1])
        
    def get_z_cost(self, img): 
        return self.state_estimator(self.state_encoder_cost(img)[1])
    
    def get_z_action(self, img): 
        return self.action_estimator(self.action_encoder(img)[0])
    
    def get_z(self, img): 
        return self.state_encoder(img)

class BisimEncoder_Head_BP_Value_Net(nn.Module):
    def __init__(self, hidden_dim, output_dim, causal=False) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lidar_dim = 240
        self.ego_state_dim = 19
        self.other_state_dim = 16
        
        self.causal = causal
        # self.state_encoder = StateEncoder(obs_dim=35, hidden_dim=hidden_dim)
        if causal:
            self.value_encoder = ImageStateEncoder_Indiv(hidden_dim=hidden_dim, nc=5, output_dim=hidden_dim)
            self.lidar_encoder = StateEncoder(self.lidar_dim, hidden_dim)
            self.state_encoder = StateEncoder(self.ego_state_dim+self.other_state_dim, hidden_dim)
                                    
            # self.value_encoder = ImageStateEncoder(hidden_dim=hidden_dim, nc=5, output_dim=hidden_dim)
            # self.value_encoder_cost = ImageStateEncoder(hidden_dim=hidden_dim, nc=5, output_dim=hidden_dim)
            
        else:
            self.state_encoder = ImageStateEncoder_NonCausal(hidden_dim=hidden_dim, nc=5, output_dim=hidden_dim)
        
        
        self.agg_reward = nn.Sequential(*[nn.Linear(hidden_dim*3, hidden_dim), nn.Tanh()])
        self.agg_cost = nn.Sequential(*[nn.Linear(hidden_dim*5, hidden_dim), nn.Tanh()])
        
        self.value_estimator = nn.Sequential(*[
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(), 
            nn.Linear(hidden_dim, 1)  
        ])
        
        self.value_cost_estimator = nn.Sequential(*[
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(), 
            nn.Linear(hidden_dim, 1)  
        ])
        
        
        self.criterion_est = nn.MSELoss()
        self.criterion_val = nn.MSELoss('mean')

        # self.opt_actor = optim.Adam(self.parameters(), lr=1e-4, betas=(0.9, 0.999))
        # self.opt_critic = optim.Adam(self.parameters(), lr=1e-4, betas=(0.9, 0.999))
        self.sigma_min = 1e-2
        self.sigma_max = 1.
        
        self.state_dynamics = ProbabilisticTransitionModel(input_dim=35, encoder_feature_dim=64, action_shape=(2, ), layer_width=64)
        self.state_dynamics.load_state_dict(torch.load('/home/haohong/0_causal_drive/baselines_clean/transition_dynamics/transition_model_full.pt',\
                                                        map_location="cpu"
                                                        ))        
    def _compute_bisim(self, z, state, action, r):
        batch_size = state.shape[0]
        perm_idx = np.random.permutation(batch_size)

        mu_state_next, sigma_state_next = self.state_dynamics(state, action)
        mu_state_next_2 = mu_state_next.detach()[perm_idx]
        sigma_state_next_2 = sigma_state_next.detach()[perm_idx]

        
        ### task 1: (causal) state estimation

        ### task 2: cost bisimulation metrics
        z_2 = z[perm_idx]
        r_2 = r[perm_idx]
        
        z_dist = F.smooth_l1_loss(z, z_2)
        r_dist = F.smooth_l1_loss(r.float(), r_2.float())
        trans_dist = torch.sqrt((mu_state_next.detach()-mu_state_next_2).pow(2) + (
                                sigma_state_next.detach()-sigma_state_next_2).pow(2))

        bisimilarity = r_dist + 0.99 * trans_dist

        loss_bisim = (z_dist - bisimilarity).pow(2).mean()

        return loss_bisim        


    def forward(self, img, lidar, state, action, reward, cost, vr_target, vc_target, deterministic=False):
        '''
            Get the cost-aware Bisimulation Loss for the representation
        '''
        if self.causal:
            z_img = self.value_encoder(img)
            z_lidar = self.lidar_encoder(lidar)
            z_ego = self.state_encoder(state)
            # print(z_img.shape, z_lidar.shape, z_ego.shape)
            
            z_reward = torch.cat([z_img[:, :self.hidden_dim*2], z_ego], dim=-1)
            z_cost = torch.cat([z_img[:, self.hidden_dim*2:], z_lidar, z_ego], dim=-1)
            # print(z_reward.shape, z_cost.shape)
            
            z_reward = self.agg_reward(z_reward)
            z_cost = self.agg_cost(z_cost)
            
            pred_vr = self.value_estimator(z_reward)
            loss_vr = self.criterion_val(pred_vr, vr_target)
            
            pred_vc = self.value_cost_estimator(z_cost)
            loss_vc = self.criterion_val(pred_vc, vc_target)
            
            loss_est = torch.tensor(0.)
            loss_act = torch.tensor(0.)
            # loss_vc = torch.tensor(0.)
            
            loss_bisim_r = self._compute_bisim(z_reward, state, action, reward)
            loss_bisim_c = self._compute_bisim(z_cost, state, action, cost)
            
            return loss_est, loss_act, loss_vr, loss_vc, loss_bisim_r, loss_bisim_c
        
        else:
            raise ValueError('none')

            return loss_est, loss_cls, pred_cls

        # return state_next_est, z, kl_loss, loss_bisim, loss_bisim_cost

    def get_z_reward(self, img): 
        return self.state_estimator(self.state_encoder_reward(img)[1])
        
    def get_z_cost(self, img): 
        return self.state_estimator(self.state_encoder_cost(img)[1])
    
    def get_z_action(self, img): 
        return self.action_estimator(self.action_encoder(img)[0])
    
    def get_z(self, img): 
        return self.state_encoder(img)

def get_train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", help="train/test")
    parser.add_argument("--model", type=str, default="encoder.pt", help="checkpoint to load")
    parser.add_argument("--encoder", type=str, default="image", help="image/state")
    parser.add_argument("--causal", type=bool, default=False, help="causal encoder")


    return parser

if __name__ == '__main__':
    from utils.dataset import *
    args = get_train_parser().parse_args()

    if args.encoder == 'image':
        train_set = BisimDataset(file_path='/home/haohong/0_causal_drive/baselines_clean/data/data_bisim_generalize_post', num_files=int(NUM_FILES*0.8), balanced=True) # TODO: //10
        val_set = BisimDataset(file_path='/home/haohong/0_causal_drive/baselines_clean/data/data_bisim_generalize_post', \
                                num_files=NUM_FILES-int(NUM_FILES*0.8), offset=int(NUM_FILES*0.8), noise_scale=0.0)
        val_set_noise = BisimDataset(file_path='../data/data_bisim_generalize_post', \
                                num_files=NUM_FILES-int(NUM_FILES*0.8), offset=int(NUM_FILES*0.8), noise_scale=0.5)
    
    elif args.encoder == 'image_spurious': 
        train_set = BisimDataset_Fusion_Spurious(file_path='/home/haohong/0_causal_drive/baselines_clean/data/data_bisim_cost_continuous_post', noise_scale=20, num_files=int(NUM_FILES*0.8), balanced=True) # TODO: //10
        val_set = BisimDataset_Fusion_Spurious(file_path='/home/haohong/0_causal_drive/baselines_clean/data/data_bisim_cost_continuous_post', \
                                num_files=NUM_FILES-int(NUM_FILES*0.8), offset=int(NUM_FILES*0.8), noise_scale=0)
        val_set_noise = BisimDataset_Fusion_Spurious(file_path='/home/haohong/0_causal_drive/baselines_clean/data/data_bisim_cost_continuous_post', \
                                num_files=NUM_FILES-int(NUM_FILES*0.8), offset=int(NUM_FILES*0.8), noise_scale=20)
    
    elif args.encoder == 'state':
        train_set = BisimDataset(file_path='/home/haohong/0_causal_drive/baselines_clean/data/data_bisim_generalize_post', num_files=50000, balanced=True) # TODO: //10
        val_set = BisimDataset(file_path='/home/haohong/0_causal_drive/baselines_clean/data/data_bisim_generalize_post', \
                                            num_files=50000, offset=2*NUM_FILES//10, balanced=True)
        val_set_noise = val_set

    else: 
        raise ValueError(args.encoder + ' Not Implemented')
    
    train_dataloader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=16)
    val_dataloader = DataLoader(val_set, batch_size=1024, shuffle=True, num_workers=16)
    val_dataloader_noise = DataLoader(val_set_noise, batch_size=1024, shuffle=True, num_workers=16)
    
    criterion = nn.MSELoss(reduction='mean')

    if args.encoder in ['image', 'image_spurious']:
        model = BisimEncoder_Head_BP_Value_Net(hidden_dim=64, output_dim=35, causal=args.causal)
        
    else:
        pass
        # model = BisimEncoder_State(hidden_dim=64, output_dim=2, causal=args.causal)

    model = CUDA(model)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
    optimizer_c = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
        
    best_val_loss = 1e9

    if args.mode == 'train': 
        NUM_EPOCHS = 100
        if args.encoder in ['image', 'image_spurious']:
            for epoch in range(NUM_EPOCHS):
                loss_train, loss_val, loss_test = 0, 0, 0
                print('===========================')
                model.train()
                for data in train_dataloader:
                    img, _, lidar, _, state, state_next, action, reward, cost, vr_target, vc_target = data
                    img = CUDA(img)
                    lidar = CUDA(lidar)
                    
                    state = CUDA(state)
                    state_next = CUDA(state_next)
                    action = CUDA(action)
                    reward = CUDA(reward)
                    cost = CUDA(torch.LongTensor(cost.reshape(-1)))
                    vr_target = (CUDA(vr_target)-40) / 40.
                    vc_target = CUDA(vc_target) / 10.
                    
                    if args.causal: 
                        # model(img, lidar, state, action, reward, cost, vr_target, vc_target)
                        loss_est, loss_act, loss_vr,  loss_vc, loss_bisim, loss_bisim_cost = \
                            model(img, lidar, state, action, reward, cost, vr_target, vc_target)
                    else: 
                        loss_est, loss_cls,  pred_cls = model(img, state, action, reward, cost)

                    loss_norm = 0.
                    for w in model.parameters(): 
                        loss_norm += w.norm(2)
                    
                    loss_reward = loss_vr + 0.01*loss_bisim #  loss_est + #  loss_est + loss_act # + loss_cls + loss_bisim + loss_bisim_cost # loss_state_est + 0.1*kl_loss + loss_bisim_cost
                    loss_cost = loss_vc + 0.01*loss_bisim_cost
                    # loss += 0.0001 * loss_norm
                    optimizer.zero_grad()
                    optimizer_c.zero_grad()
                    loss_reward.backward(retain_graph=True)
                    loss_cost.backward()
                    optimizer.step()
                    optimizer_c.zero_grad()
                    
                    loss_train += (loss_vr + loss_vc).item()
                    loss = loss_cost + loss_reward
                    print('train loss: {:4f} | state est. loss: {:.4f} | act loss: {:.4f} | V_r loss: {:.4f} | V_c loss: {:.4f} | bisim loss: {:.4f} | bisim loss cost: {:.4f} | loss norm: {:.4f}'.format(
                        loss.item(), loss_est.item(), loss_act.item(), loss_vr.item(), loss_vc.item(), loss_bisim.item(), loss_bisim_cost.item(), loss_norm.item()), end='\r')
                
                loss_train /= len(train_dataloader)
                print('\n')
                model.eval()

                loss_val = eval_loss_dataloader_state(model, val_dataloader)
                # loss_val_noise = eval_loss_dataloader_state(model, val_dataloader_noise)
                
                print('Epoch {:3d}, train_loss: {:4f}, val_loss:  {:4f}'.format(epoch, loss_train, loss_val))
                
                if loss_val < best_val_loss:
                    print('best val loss find!')
                    best_val_loss = loss_val
                    torch.save(model.state_dict(), args.model+'.pt')
                    print('model saved!')
        else: 
            for epoch in range(NUM_EPOCHS):
                loss_train, loss_val, loss_test = 0, 0, 0
                print('===========================')
                model.train()
                for data in train_dataloader:
                    img, _, state, state_next, action, reward, cost = data
                    img = CUDA(img)
                    state = CUDA(state)
                    state_next = CUDA(state_next)
                    action = CUDA(action)
                    reward = CUDA(reward)
                    cost = CUDA(torch.LongTensor(cost.reshape(-1)))

                    if args.causal: 
                        loss_est, loss_cls,  pred_cls, loss_bisim, loss_bisim_cost = model(img, state, action, reward, cost)
                    else: 
                        loss_est, loss, pred_cls = model(img, state, action, reward, cost)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss_train += loss.item()
                    print('train loss: {:4f}'.format(loss.item()), end='\r')
                
                loss_train /= len(train_dataloader)
                print('\n')
                model.eval()

                loss_val = eval_loss_dataloader_state(model, val_dataloader)
                
                print('Epoch {:3d}, train_loss: {:4f}, val_loss:  {:4f}'.format(epoch, loss_train, loss_val))

                if loss_val < best_val_loss:
                    print('best val loss find!')
                    best_val_loss = loss_val
                    torch.save(model.state_dict(), args.model+'.pt')
                    print('model saved!')   
    else:   # mode = test
        if args.encoder in ['image', 'image_spurious']:
            visualize_ckpt(model, args.model+'.pt', val_dataloader)
        else:
            visualize_ckpt_state(model, args.model+'.pt', val_dataloader)
