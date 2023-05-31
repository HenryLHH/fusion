import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
# from .transition_model import ProbabilisticTransitionModel

torch.autograd.set_detect_anomaly(True)

OBS_DIM = 35
NUM_FILES = 390000

def CPU(x):
    return x.detach().cpu().numpy()

def CUDA(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.cuda()

def reparameterize(mu, std):
    # std = torch.exp(logstd)
    eps = torch.randn_like(std)
    return mu + eps * std

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)
    return total_kld, dimension_wise_kld, mean_kld


def eval_loss_dataloader_state(model, dataloader, verbose=0):
    model.eval()
    loss = 0
    accuracy = 0.0
    recall_all, precision_all = 0.0, 0.0

    pbar = tqdm(total=len(dataloader))
    for data in dataloader:
        pbar.update(1)
        img, _, lidar, _, state, state_next, action, reward, cost, vr_target, vc_target, _ = data
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
        _, loss_state = model(img, state, deterministic=True)

        loss += (loss_state).item() # (loss_est+loss_ret+loss_bisim+loss_bisim_cost).item()
        # print('train loss: {:4f} | precision: {:.4f} | recall: {:.4f} | acc: {:.4f} '.format(loss_ret.item(), precision, recall, acc), end='\r')
        torch.cuda.empty_cache()
    
    loss /= len(dataloader)
    recall_all /= len(dataloader)
    precision_all /= len(dataloader)
    accuracy /= len(dataloader)
    print('state est. loss: {:.4f} '.format(loss))

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

    def forward(self, x): 

        z_tmp = self.encoder(x)
        mu, std = z_tmp[:, :self.hidden_dim*self.nc], z_tmp[:, self.hidden_dim*self.nc:]
        z_tmp = reparameterize(mu, std)
        
        output = self.fc_out(z_tmp)

        mu, std = output[:, :self.output_dim], output[:, self.output_dim:]
        state_pred = reparameterize(mu, std)

        return state_pred, mu, torch.exp(std)


class BisimEncoder_Head_BP_Actor(nn.Module):
    def __init__(self, hidden_dim, output_dim, causal=False) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lidar_dim = 240
        self.causal = causal
        # self.state_encoder = StateEncoder(obs_dim=35, hidden_dim=hidden_dim)
        if causal:
            self.action_encoder_ego = ImageStateEncoder(hidden_dim=hidden_dim, nc=2, output_dim=19)
            self.action_encoder_surrounding = ImageStateEncoder(hidden_dim=hidden_dim, nc=4, output_dim=16)
            
        else:
            self.action_encoder = ImageStateEncoder_NonCausal(hidden_dim=hidden_dim, nc=5, output_dim=output_dim)
        
        self.criteria = nn.MSELoss(reduction='mean')        
        
        # self.opt_actor = optim.Adam(self.parameters(), lr=1e-4, betas=(0.9, 0.999))
        # self.opt_critic = optim.Adam(self.parameters(), lr=1e-4, betas=(0.9, 0.999))
        self.sigma_min = 1e-2
        self.sigma_max = 1.
    
    def forward(self, img, state, deterministic=False):
        '''
            Get the cost-aware Bisimulation Loss for the representation
        '''
        if self.causal: 
            if not deterministic: 
                state_pred_ego, _, _ = self.action_encoder_ego(img[:, :2, :, :])
                state_pred_context, _, _ = self.action_encoder_surrounding(img[:, 1:, :, :])
                
            else:
                _, state_pred_ego, _ = self.action_encoder_ego(img[:, :2, :, :])
                _, state_pred_context, _ = self.action_encoder_surrounding(img[:, 1:, :, :])
            state_pred = torch.cat([state_pred_ego, state_pred_context], axis=1)
            
        else: 
            if not deterministic: 
                state_pred, _, _ = self.action_encoder(img)
            else:
                _, state_pred, _ = self.action_encoder(img)

        loss_state = self.criteria(state_pred, state)
        
        return state_pred, loss_state
    

class ICILEncoder_Head_BP_Actor(nn.Module):
    def __init__(self, hidden_dim, output_dim, causal=False) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lidar_dim = 240
        self.causal = causal
        # self.state_encoder = StateEncoder(obs_dim=35, hidden_dim=hidden_dim)
        if causal:
            self.action_encoder = ImageStateEncoder(hidden_dim=hidden_dim, nc=5, output_dim=hidden_dim)
        else:
            self.action_encoder = ImageStateEncoder_NonCausal(hidden_dim=hidden_dim, nc=5, output_dim=hidden_dim)
        
        self.action_estimator = nn.Sequential(*[
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, 4)  
        ])
        
        self.criterion_est = nn.MSELoss()
        self.criterion_val = nn.MSELoss('mean')
        
        # self.opt_actor = optim.Adam(self.parameters(), lr=1e-4, betas=(0.9, 0.999))
        # self.opt_critic = optim.Adam(self.parameters(), lr=1e-4, betas=(0.9, 0.999))
        self.sigma_min = 1e-2
        self.sigma_max = 1.
    
    def forward(self, img, lidar, state, action, reward, cost, vr_target, vc_target, deterministic=False):
        '''
            Get the cost-aware Bisimulation Loss for the representation
        '''

        if not deterministic: 
            z_act, _, _ = self.action_encoder(img)
        else:
            _, z_act, _ = self.action_encoder(img)
        
        a_pred = self.action_estimator(z_act)
        if not deterministic: 
            mu_act, std_act = a_pred[:, :2], a_pred[:, 2:]
            # std_act = torch.clamp(std_act, np.log(self.sigma_min), np.log(self.sigma_max)).exp()

            std_act = torch.sigmoid(std_act)*(self.sigma_max-self.sigma_min) + self.sigma_min
            act = reparameterize(mu_act, std_act)
        else: 
            act = a_pred[:, :2]
        
        loss_act = self.criterion_est(act, action)
        
        loss_est = torch.tensor(0.)
        loss_vr = torch.tensor(0.)
        loss_vc = torch.tensor(0.)
        loss_bisim_r = torch.tensor(0.) # self._compute_bisim(z_reward, state, action, reward)
        loss_bisim_c = torch.tensor(0.) # self._compute_bisim(z_cost, state, action, cost)
        
        return loss_est, loss_act, loss_vr, loss_vc, loss_bisim_r, loss_bisim_c
    

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
        train_set = BisimDataset(file_path='../data/data_bisim_generalize_post', num_files=int(NUM_FILES*0.8), balanced=True) # TODO: //10
        val_set = BisimDataset(file_path='../data/data_bisim_generalize_post', \
                                num_files=NUM_FILES-int(NUM_FILES*0.8), offset=int(NUM_FILES*0.8), noise_scale=0.0)
        val_set_noise = BisimDataset(file_path='../data/data_bisim_generalize_post', \
                                num_files=NUM_FILES-int(NUM_FILES*0.8), offset=int(NUM_FILES*0.8), noise_scale=0.5)
    
    elif args.encoder in ['image_spurious', 'icil']: 
        train_set = BisimDataset_Fusion_Spurious(file_path='/home/haohong/0_causal_drive/baselines_clean/envs/data_mixed_dynamics_post', noise_scale=0, num_files=int(NUM_FILES*0.8), balanced=True) # TODO: //10
        val_set = BisimDataset_Fusion_Spurious(file_path='/home/haohong/0_causal_drive/baselines_clean/envs/data_mixed_dynamics_post', \
                                num_files=NUM_FILES-int(NUM_FILES*0.8), offset=int(NUM_FILES*0.8), noise_scale=0)
        val_set_noise = BisimDataset_Fusion_Spurious(file_path='/home/haohong/0_causal_drive/baselines_clean/envs/data_mixed_dynamics_post', \
                                num_files=NUM_FILES-int(NUM_FILES*0.8), offset=int(NUM_FILES*0.8), noise_scale=20)
    
    elif args.encoder == 'state':
        train_set = BisimDataset(file_path='../data/data_bisim_generalize_post', num_files=50000, balanced=True) # TODO: //10
        val_set = BisimDataset(file_path='../data/data_bisim_generalize_post', \
                                            num_files=50000, offset=2*NUM_FILES//10, balanced=True)
        val_set_noise = val_set

    else: 
        raise ValueError(args.encoder + ' Not Implemented')
    
    train_dataloader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=16)
    val_dataloader = DataLoader(val_set, batch_size=1024, shuffle=True, num_workers=16)
    val_dataloader_noise = DataLoader(val_set_noise, batch_size=1024, shuffle=True, num_workers=16)
    
    criterion = nn.MSELoss(reduction='mean')

    if args.encoder in ['image', 'image_spurious']:
        model = BisimEncoder_Head_BP_Actor(hidden_dim=64, output_dim=35, causal=args.causal)
        
    elif args.encoder in ['icil']:
        model = ICILEncoder_Head_BP_Actor(hidden_dim=64, output_dim=35, causal=args.causal)
    else:
        raise ValueError('encoder {} not supported'.format(args.encoder))

    model = CUDA(model)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4)
        
    best_val_loss = 1e9

    if args.mode == 'train': 
        NUM_EPOCHS = 100
        if args.encoder in ['image', 'image_spurious', 'icil']:
            for epoch in range(NUM_EPOCHS):
                loss_train, loss_val, loss_test = 0, 0, 0
                print('===========================')
                model.train()
                for data in train_dataloader:
                    img, _, lidar, _, state, state_next, action, reward, cost, vr_target, vc_target, _ = data
                    img = CUDA(img)
                    lidar = CUDA(lidar)
                    
                    state = CUDA(state)
                    state_next = CUDA(state_next)
                    action = CUDA(action)
                    reward = CUDA(reward)
                    cost = CUDA(torch.LongTensor(cost.reshape(-1)))
                    vr_target = (CUDA(vr_target)-40) / 40.
                    vc_target = CUDA(vc_target) / 10.
                    
                    state_est, loss_state = model.forward(img, state)

                    loss_norm = 0.
                    for w in model.parameters(): 
                        loss_norm += w.norm(2)
                    
                    loss = loss_state #  loss_est + #  loss_est + loss_act # + loss_cls + loss_bisim + loss_bisim_cost # loss_state_est + 0.1*kl_loss + loss_bisim_cost
                    # loss += 0.0001 * loss_norm
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    loss_train += loss.item()
                    print('state est. loss: {:.4f}'.format(
                        loss.item()), end='\r')
                
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
        if args.encoder in ['image', 'image_spurious', 'icil']:
            visualize_ckpt(model, args.model+'.pt', val_dataloader)
        else:
            visualize_ckpt_state(model, args.model+'.pt', val_dataloader)
