import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import argparse

def get_train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="encoder", help="checkpoint to load")    

    return parser


args = get_train_parser().parse_args()
filenames = glob.glob('log/'+args.model)
print(filenames)
data = [np.load(f, allow_pickle=True) for f in filenames]

plt.figure(figsize=(8, 18))
plt.subplot(311)
for d in data:
    plt.plot(d.item()['reward'])
    print(np.max(d.item()['reward']))
plt.subplot(312)
for d in data:
    plt.plot(d.item()['cost'])
    print(np.min(d.item()['cost']))
    
plt.subplot(313)
for d in data:
    plt.plot(d.item()['success_rate'])
    print(np.max(d.item()['success_rate']))

plt.legend([f.split('/')[-1].split('.')[0] for f in filenames])
plt.savefig('plot_curves')