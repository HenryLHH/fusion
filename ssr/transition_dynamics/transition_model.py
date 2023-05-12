import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

OBS_DIM = 35
ACT_DIM = 2
NUM_FILES = 400000
NUM_EPOCHS = 100


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

def normc_initializer(std: float = 1.0):
    def initializer(tensor):
        tensor.data.normal_(0, 1)
        tensor.data *= std / torch.sqrt(tensor.data.pow(2).sum(1, keepdim=True))

    return initializer

class StateEncoder(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, hidden_dim=16):
        super().__init__()
        self.encoder_1 = nn.Linear(obs_dim, hidden_dim)
        self.encoder_2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):        
        x = self.encoder_1(x)
        x = torch.relu(x)
        x = self.encoder_2(x)
        return x

class ProbabilisticTransitionModel(nn.Module):
    
    def __init__(self, input_dim, encoder_feature_dim, action_shape, layer_width, announce=True, max_sigma=1e1, min_sigma=1e-4):
        super().__init__()
        self.input_encoder = StateEncoder(input_dim, encoder_feature_dim)
        self.fc = nn.Linear(encoder_feature_dim + action_shape[0], layer_width)
        self.ln = nn.LayerNorm(layer_width)
        self.fc_mu = nn.Linear(layer_width, input_dim)
        self.fc_sigma = nn.Linear(layer_width, input_dim)
        initializer = normc_initializer(1.0)
        initializer(self.input_encoder.encoder_1.weight)
        initializer(self.input_encoder.encoder_2.weight)
        initializer(self.fc.weight)
        initializer(self.fc_mu.weight)
        initializer(self.fc_sigma.weight)
        
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        assert(self.max_sigma >= self.min_sigma)
        if announce:
            print("Probabilistic transition model chosen.")

    def forward(self, s, a):
        z = self.encode(s)
        return self.predict(z, a)

    def sample_prediction(self, s, a):
        mu, sigma = self(s, a)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps

    def encode(self, s):
        return self.input_encoder(s)
    
    def predict(self, z, a):
        x = torch.cat([z, a], dim=-1)
        x = self.fc(x)
        x = self.ln(x)
        x = torch.relu(x)

        mu = self.fc_mu(x)
        sigma = torch.sigmoid(self.fc_sigma(x))  # range (0, 1.)
        sigma = self.min_sigma + (self.max_sigma - self.min_sigma) * sigma  # scaled range (min_sigma, max_sigma)
        return mu, sigma

if __name__ == '__main__':
    from dataset import *
    # dataset = TransitionDataset(num_files=NUM_FILES)
    dataset = TransitionDataset_35(num_files=NUM_FILES)

    num_training, num_testing = int(NUM_FILES*0.8), NUM_FILES-int(NUM_FILES*0.8)
    
    train_set, val_set = torch.utils.data.random_split(dataset, [num_training, num_testing])
    train_dataloader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=16)
    val_dataloader = DataLoader(val_set, batch_size=1024, shuffle=True, num_workers=16)
    criterion = nn.MSELoss(reduction='mean')
    
    model = ProbabilisticTransitionModel(input_dim=OBS_DIM, encoder_feature_dim=64, action_shape=(ACT_DIM,), layer_width=64)
    model = CUDA(model)

    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
    print(len(train_dataloader))
    for epoch in range(NUM_EPOCHS):
        loss_train, loss_val = 0, 0
        print('===========================')
        for s, a, s_next in train_dataloader:
            s = Variable(s.cuda())
            a = Variable(a.cuda())
            s_next = CUDA(s_next)
            
            s_next_pred = model.sample_prediction(s, a)

            loss = criterion(s_next_pred, s_next)  
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            print('train loss: {:4f}'.format(loss.item()), end='\r')

        loss_train /= len(train_dataloader)
        print('\n')
        for s, a, s_next in val_dataloader:
            s = Variable(s.cuda())
            a = Variable(a.cuda())
            s_next = CUDA(s_next)
            s_next_pred, s_next_var  = model(s, a)
            loss_val += criterion(s_next_pred, s_next).item()
        #     samplewise_loss = torch.square(s_next_pred-s_next).mean(-1)
        # plt.figure()
        # plt.hist(samplewise_loss.detach().cpu().numpy(), bins=30)
        # plt.savefig('demo.png')
        # plt.close()
        loss_val /= len(val_dataloader)
        print(CPU(s[0]), CPU(a[0]))
        print('truth: ', CPU(s_next[0]))
        print('pred: ', CPU(s_next_pred[0]))
        print('var: ', CPU(s_next_var[0]))
        print('Epoch {:3d}, train_loss: {:4f}, val loss:  {:4f}'.format(epoch, loss_train, loss_val))
    
    torch.save(model.state_dict(), 'transition_model_full.pt')
    