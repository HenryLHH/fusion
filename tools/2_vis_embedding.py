import torch
import argparse

from envs.envs import State_TopDownMetaDriveEnv

from ssr.agent.DT.utils import evaluate_on_env_structure
from ssr.agent.DT.model import SafeDecisionTransformer_Structure
from utils.utils import CUDA

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

def get_train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="encoder", help="checkpoint to load")
    parser.add_argument("--seed", type=int, default=0, help="checkpoint to load")

    return parser



if __name__ == '__main__': 
    args = get_train_parser().parse_args()
    
    model = CUDA(SafeDecisionTransformer_Structure(state_dim=35, act_dim=2, n_blocks=3, h_dim=64, context_len=30, n_heads=4, drop_p=0.1, max_timestep=1000))

    model.load_state_dict(torch.load('checkpoint/'+args.model+'.pt'))
    print('model loaded')
    rtg = CUDA(torch.arange(0, 7.5, 0.25).reshape(-1, 1))
    ctg = CUDA(torch.arange(0, 2.0, 0.068).reshape(-1, 1))

    rtg_embedding = model.embed_rtg(rtg)
    ctg_embedding = model.embed_rtg(ctg)
    print(rtg_embedding.shape, ctg_embedding.shape)
    data_viz = torch.cat([rtg_embedding, ctg_embedding], dim=-1)

    model_viz_r = TSNE(n_components=2, perplexity=5)
    model_viz_c = TSNE(n_components=2, perplexity=5)
    model_viz_mixed = TSNE(n_components=2, perplexity=5)
    
    rtg_viz = model_viz_r.fit_transform(rtg_embedding.detach().cpu().numpy())
    ctg_viz = model_viz_c.fit_transform(ctg_embedding.detach().cpu().numpy())
    data_viz = model_viz_mixed.fit_transform(data_viz.detach().cpu().numpy())
    
    color_r = ['#{:02d}{:02d}{:02d}'.format(3*i, 3*i, 3*i) for i in range(30)]
    color_c = ['#{:02d}{:02d}{:02d}'.format(2*i, 2*i, 2*i) for i in range(30)]
    
    plt.figure(figsize=(6, 15))
    plt.subplot(311)
    for i in range(30): 
        plt.scatter(rtg_viz[i, 0], rtg_viz[i, 1], c=color_r[i], s=i)
        plt.title('Reward-to-go Embeddings')
    plt.subplot(312)
    for i in range(30): 
        plt.scatter(ctg_viz[i, 0], ctg_viz[i, 1], c=color_c[i], s=i)
        plt.title('Cost-to-go Embeddings')
    
    plt.subplot(313)
    for i in range(30):
        plt.scatter(data_viz[i, 0], data_viz[i, 1], c=color_c[i], s=i)
        plt.title('Concatenated Embeddings')
    
    plt.savefig('tsne')