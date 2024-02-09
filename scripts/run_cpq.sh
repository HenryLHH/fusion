#!/bin/bash
python train/train_cpq.py --eval_every 2500 --cost_limit 1 \
        --dataset dataset_mixed_single_post --single_env True --eval_episodes 50