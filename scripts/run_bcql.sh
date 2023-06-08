#!/bin/bash
python train/train_bearl.py --eval_every 2500 --cost_limit 10 \
        --dataset dataset_mixed_post
# python train/train_dt_single_env.py --rtg_scale 400.0 --ctg_scale 10.0 \
#     --dataset data_mixed_dynamics --model context_20_repeat --context 20 
