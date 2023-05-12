import torch
import torch.nn as nn
import numpy as np

OBS_DIM = 19
NUM_FILES = 400000
def CPU(x):
    return x.detach().cpu().numpy()

def CUDA(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.cuda()


def eval_loss_dataloader(model, criterion, dataloader, verbose=0):

    loss = 0
    for img, state in dataloader: 
        img = CUDA(img)
        state = CUDA(state)            
        state_pred, mu, std = model(img)
        loss += criterion(mu, state).item()
    loss /= len(dataloader)
    if verbose: 
        print('truth: ', CPU(state[0]))
        print('pred: ', CPU(mu[0]))
        print('var: ', CPU(std[0]))
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
    
class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)



class ImageStateEncoder_NonCausal(nn.Module):
    def __init__(self, hidden_dim=16, nc=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.nc = nc
        
        self.encoder = build_encoder_net(hidden_dim*nc, nc)
        self.fc_out = nn.Linear(hidden_dim*nc, OBS_DIM*2)
        self.ln = nn.LayerNorm(OBS_DIM)

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x): 

        z_tmp = self.encoder(x)
        mu, std = z_tmp[:, :self.hidden_dim*self.nc], z_tmp[:, self.hidden_dim*self.nc:]
        z_tmp = self.reparameterize(mu, std)
        
        output = self.fc_out(z_tmp)

        mu, std = output[:, :OBS_DIM], output[:, OBS_DIM:]
        state_pred = self.reparameterize(mu, std)

        return state_pred, mu, torch.exp(std)

class ImageStateEncoder(nn.Module):
    def __init__(self, hidden_dim=16, nc=3, output_dim=OBS_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.nc = nc
        self.output_dim = output_dim

        self.encoder = nn.ModuleList([build_encoder_net(hidden_dim, 1) for _ in range(3)])
        self.fc_out = nn.Linear(hidden_dim*nc, output_dim*2)
        self.ln = nn.LayerNorm(output_dim)

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x): 
        mu, std = [], []
        z = []
        for i in range(len(self.encoder)):
            z_tmp = self.encoder[i](x[:, [i], :, :])
            mu_tmp, std_tmp = z_tmp[:, :self.hidden_dim], z_tmp[:, self.hidden_dim:]
            mu.append(mu_tmp)
            std.append(std_tmp)
        
            z_tmp = self.reparameterize(mu_tmp, std_tmp)
            z.append(z_tmp)
        z = torch.cat(z, dim=-1)
        output = self.fc_out(z)

        mu, std = output[:, :self.output_dim], output[:, self.output_dim:]
        state_pred = self.reparameterize(mu, std)

        return state_pred, mu, torch.exp(std)

class StateEncoder(nn.Module):
    def __init__(self, hidden_dim=16, output_dim=16):
        super().__init__()
        self.output_dim = output_dim
        self.encoder = nn.Sequential(
            *[nn.Linear(35, hidden_dim), 
              nn.ReLU(), 
              nn.Linear(hidden_dim, hidden_dim),
              nn.ReLU(),
              nn.Linear(hidden_dim, output_dim*2)]
        )
    
    def forward(self, s):
        output = self.encoder(s)
        mu = output[:, :self.output_dim]
        log_std = output[:, self.output_dim:]

        return mu, torch.exp(log_std)


if __name__ == '__main__':
    from dataset import *

    NUM_EPOCHS = 1

    train_set = ImageTransitionDataset_Gen(file_path='../envs/data_bisim_generalize_post', num_files=NUM_FILES//10)
    val_set = ImageTransitionDataset_Gen(file_path='../envs/data_bisim_generalize_post', \
                                            num_files=NUM_FILES-NUM_FILES//10, offset=0*NUM_FILES//10, noise_scale=1.0)
    val_set_raw = ImageTransitionDataset_Gen(file_path='../envs/data_bisim_generalize_post', \
                                            num_files=NUM_FILES-NUM_FILES//10, offset=0*NUM_FILES//10, noise_scale=0.0)
    
    test_set = ImageTransitionDataset(file_path='../VAE/data_bisim_post' , num_files=NUM_FILES, noise_scale=1.0)
    test_set_raw = ImageTransitionDataset(file_path='../VAE/data_bisim_post' , num_files=NUM_FILES, noise_scale=0.0)

    # num_training, num_testing = int(NUM_FILES*0.8), NUM_FILES-int(NUM_FILES*0.8)
    
    # train_set, val_set = torch.utils.data.random_split(dataset, [num_training, num_testing])
    train_dataloader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=16)
    val_dataloader = DataLoader(val_set, batch_size=128, shuffle=True, num_workers=16)
    test_dataloader = DataLoader(test_set, batch_size=128, shuffle=True, num_workers=16)
    val_dataloader_raw = DataLoader(val_set_raw, batch_size=128, shuffle=True, num_workers=16)
    test_dataloader_raw = DataLoader(test_set_raw, batch_size=128, shuffle=True, num_workers=16)

    criterion = nn.MSELoss(reduction='mean')

    # model = ImageStateEncoder_NonCausal(hidden_dim=16, nc=3)
    model = ImageStateEncoder(hidden_dim=16, nc=3)

    model = CUDA(model)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
    best_val_loss = 1e9

    # model.load_state_dict(torch.load('encoder_noncausal.pt'))
    model.load_state_dict(torch.load('encoder_causal.pt'))

    for epoch in range(NUM_EPOCHS):
        loss_train, loss_val, loss_test = 0, 0, 0
        print('===========================')
        # for img, state in train_dataloader:
        #     img = Variable(CUDA(img))
        #     state = CUDA(state)
            
        #     state_pred, mu, std = model(img)
        #     loss = criterion(state_pred, state)  
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        #     loss_train += loss.item()
        #     print('train loss: {:4f}'.format(loss.item()), end='\r')

        # loss_train /= len(train_dataloader)
        # print('\n')
        model.eval()

        loss_val = eval_loss_dataloader(model, criterion, val_dataloader)
        loss_val_raw = eval_loss_dataloader(model, criterion, val_dataloader_raw)

        loss_test = eval_loss_dataloader(model, criterion, test_dataloader)
        loss_test_raw = eval_loss_dataloader(model, criterion, test_dataloader_raw)

        print('Epoch {:3d}, train_loss: {:4f}, val loss:  {:4f}, test loss:  {:4f}'.format(epoch, loss_train, loss_val, loss_test))
        print('val loss raw:  {:4f}, test loss raw:  {:4f}'.format(loss_val_raw, loss_test_raw))

        # if loss_val < best_val_loss:
        #     print('best val loss find!')
        #     best_val_loss = loss_val
        #     torch.save(model.state_dict(), 'encoder_few_noncausal.pt')
        #     print('model saved!')