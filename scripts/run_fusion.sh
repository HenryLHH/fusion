#!/bin/bash

python train/train_fusion.py --rtg_scale 300.0 --ctg_scale 10.0 \
    --dataset data_mix --model debug_bisim --context 20 \
    --num_envs 16 --num_workers 16 --dynamics --value --single_env --bisim
