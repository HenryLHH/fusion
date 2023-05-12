import argparse


def get_train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="")
    parser.add_argument("--num-gpus", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--num-cpus-per-worker", type=float, default=0.5)
    parser.add_argument("--num-gpus-per-trial", type=float, default=0.25)
    parser.add_argument("--cost-limit", type=float, default=1.0)
    parser.add_argument("--test", action="store_true")
    return parser

