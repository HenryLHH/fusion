import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import argparse
from ssr.agent.icil.icil import ICIL
from utils.dataset import BisimDataset_Fusion_Spurious
from tqdm import tqdm



OBS_DIM = 35
NUM_FILES = 400000
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

    # pbar = tqdm(total=len(dataloader))
    for data in dataloader:
        # pbar.update(1)
        img, img_next, lidar, _, state, state_next, action, reward, cost, vr_target, vc_target = data
        img = CUDA(img)
        img_next = CUDA(img_next)
        
        # state = CUDA(state)
        # state_next = CUDA(state_next)
        action = CUDA(action)
        # reward = CUDA(reward)
        # cost = CUDA(torch.LongTensor(cost.reshape(-1)))
        # vr_target = (CUDA(vr_target)-40) / 40.
        # vc_target = CUDA(vc_target) / 10.
        
        # loss_bisim, loss_bisim_cost = 0., 0.
        policy_loss, next_state_pred_loss, next_state_energy_loss, mi_loss, mine_loss = \
            model(img, action, img_next, deterministic=True)
        
        loss_act = policy_loss + next_state_energy_loss #  loss_est + #  loss_est + loss_act # + loss_cls + loss_bisim + loss_bisim_cost # loss_state_est + 0.1*kl_loss + loss_bisim_cost
        loss_rep = next_state_pred_loss + mi_loss
        # loss += 0.0001 * loss_norm
        
        loss += (loss_act+loss_rep).item()
    

        loss += (loss_act).item() # (loss_est+loss_ret+loss_bisim+loss_bisim_cost).item()
        # print('train loss: {:4f} | precision: {:.4f} | recall: {:.4f} | acc: {:.4f} '.format(loss_ret.item(), precision, recall, acc), end='\r')
        print('policy loss: {:4f} | rep. loss: {:.4f} | act loss: {:.4f} | energy loss: {:.4f} | state pred loss: {:.4f}'.format(
            loss_act.item(), loss_rep.item(), policy_loss.item(), next_state_energy_loss.item(), next_state_pred_loss.item()), end='\r')
        torch.cuda.empty_cache()
    
    loss /= len(dataloader)
    recall_all /= len(dataloader)
    precision_all /= len(dataloader)
    accuracy /= len(dataloader)

    # pbar.close()

    return loss


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

def get_train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", help="train/test")
    parser.add_argument("--model", type=str, default="encoder", help="checkpoint to load")
    parser.add_argument("--encoder", type=str, default="image", help="image/state")
    parser.add_argument("--causal", type=bool, default=False, help="causal encoder")


    return parser

if __name__ == '__main__':
    NUM_EPOCHS = 100
    
    args = get_train_parser().parse_args()
    
    train_set = BisimDataset_Fusion_Spurious(file_path='/home/haohong/0_causal_drive/baselines_clean/data/data_bisim_cost_continuous_post', noise_scale=0, num_files=int(NUM_FILES*0.8), balanced=True) # TODO: //10
    val_set = BisimDataset_Fusion_Spurious(file_path='/home/haohong/0_causal_drive/baselines_clean/data/data_bisim_cost_continuous_post', \
                            num_files=NUM_FILES-int(NUM_FILES*0.8), offset=int(NUM_FILES*0.8), noise_scale=0)
    val_set_noise = BisimDataset_Fusion_Spurious(file_path='../data/data_bisim_cost_continuous_post', \
                            num_files=NUM_FILES-int(NUM_FILES*0.8), offset=int(NUM_FILES*0.8), noise_scale=20)

    train_dataloader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=16)
    val_dataloader = DataLoader(val_set, batch_size=1024, shuffle=True, num_workers=16)
    val_dataloader_noise = DataLoader(val_set_noise, batch_size=1024, shuffle=True, num_workers=16)
    
    model = CUDA(ICIL(state_dim=5, action_dim=2, hidden_dim_input=64, hidden_dim=64))
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))

    best_val_loss = 1e9
    for epoch in range(NUM_EPOCHS):
        loss_train, loss_val, loss_test = 0, 0, 0
        print('===========================')
        model.train()
        for data in train_dataloader:
            img, img_next, lidar, _, state, state_next, action, reward, cost, vr_target, vc_target = data
            img = CUDA(img)
            img_next = CUDA(img_next)
            state = CUDA(state)
            state_next = CUDA(state_next)
            action = CUDA(action)
            # reward = CUDA(reward)
            # cost = CUDA(torch.LongTensor(cost.reshape(-1)))
            # vr_target = (CUDA(vr_target)-40) / 40.
            # vc_target = CUDA(vc_target) / 10.

            policy_loss, next_state_pred_loss, next_state_energy_loss, mi_loss, mine_loss = \
                model(img, action, img_next)
            
            loss_act = policy_loss + next_state_energy_loss #  loss_est + #  loss_est + loss_act # + loss_cls + loss_bisim + loss_bisim_cost # loss_state_est + 0.1*kl_loss + loss_bisim_cost
            loss_rep = next_state_pred_loss + mi_loss
            # loss += 0.0001 * loss_norm
            model.policy_opt.zero_grad()
            model.rep_opt.zero_grad()
            loss_act.backward(retain_graph=True)
            loss_rep.backward()
            model.policy_opt.step()
            model.rep_opt.step()
            
            loss_train += (loss_act+loss_rep).item()
            print('policy loss: {:4f} | rep. loss: {:.4f} | act loss: {:.4f} | energy loss: {:.4f} | state pred loss: {:.4f}'.format(
                loss_act.item(), loss_rep.item(), policy_loss.item(), next_state_energy_loss.item(), next_state_pred_loss.item()), end='\r')
        
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