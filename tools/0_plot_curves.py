import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import argparse

def get_train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="encoder", help="checkpoint to load")    
    parser.add_argument("--smooth", type=int, default=1, help="smooth step of curves")    

    return parser


def np_move_avg(a, n, mode="same"):
    return np.convolve(a, np.ones((n,))/n, mode=mode)

args = get_train_parser().parse_args()
filenames = glob.glob('log/'+args.model) # + glob.glob('log/icil/icil_state*')
print(filenames)
data = [np.load(f, allow_pickle=True) for f in filenames]

plt.figure(figsize=(10,8))
plt.subplot(321)
for d in data:
    x = d.item()['reward']
    x = np_move_avg(x, args.smooth)
    plt.plot(x)
    print(np.max(d.item()['reward']))

plt.subplot(322)
for d in data:
    x = d.item()['cost']
    x = np_move_avg(x, args.smooth)
    plt.plot(x)
    print(np.min(d.item()['cost']))
    
plt.subplot(323)
for d in data:
    x = d.item()['success_rate']
    x = np_move_avg(x, args.smooth)
    plt.plot(x)
    print(np.max(d.item()['success_rate']))

plt.subplot(324)
for d in data:
    x = d.item()['oor_rate']
    x = np_move_avg(x, args.smooth)
    plt.plot(x)
    print(np.max(d.item()['oor_rate']))

plt.subplot(325)
for d in data:
    x = d.item()['crash_rate']
    x = np_move_avg(x, args.smooth)
    plt.plot(x)
    print(np.max(d.item()['crash_rate']))


plt.subplot(326)
for d in data:
    x = d.item()['crash_rate']
    x = np_move_avg(x, args.smooth)
    plt.plot(x)
    print(np.max(d.item()['crash_rate']))
plt.legend([f.split('/')[-1].split('.')[0] for f in filenames])
plt.savefig('plot_curves')