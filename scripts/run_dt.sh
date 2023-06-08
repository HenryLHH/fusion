#!/bin/bash
python train/train_dt_single_env.py --rtg_scale 300.0 --ctg_scale 80.0 \
    --dataset dataset_mixed --model context_20_dynamics_value_repeat_small_portion --context 20 \
    --num_workers 8 --dynamics --value
# python train/train_dt_single_env.py --rtg_scale 400.0 --ctg_scale 10.0 \
#     --dataset data_mixed_dynamics --model context_20_repeat --context 20 
