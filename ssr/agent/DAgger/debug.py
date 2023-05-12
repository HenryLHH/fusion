import torch
import torch.distributions as D

mu_img = torch.rand(16, 1, 2)
std_img = torch.rand(16, 1, 2)*0.1
mu_state = torch.rand(16, 1, 2)+10
std_state = torch.rand(16, 1, 2)*0.1

mu = torch.cat([mu_img, mu_state], dim=1)
std = torch.cat([std_img, std_state], dim=1)

# weighted bivariate normal distributions
mix = D.Categorical(torch.ones(16, 2))
comp = D.Independent(D.Normal(
        mu, std), 1)

gmm = D.MixtureSameFamily(mix, comp)
idx = torch.as_tensor(comp.variance[:, 0, :].max(-1)[0] > comp.variance[:, 1, :].max(-1)[0], dtype=torch.long)
idx_range = torch.arange(0, 16, 1)
print(idx_range)
print(mu[idx_range, idx, :].shape)
# print(D.Normal(mu, std).rsample().shape)