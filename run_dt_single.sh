#!/bin/bash
python scripts/train_dt_single_env.py --rtg_scale 300.0 --ctg_scale 40.0 \
    --dataset dataset_mixed_single --model context_20_single_scale --context 20 \
    --num_workers 8 --single_env
# python scripts/train_dt_single_env.py --rtg_scale 400.0 --ctg_scale 10.0 \
#     --dataset data_mixed_dynamics --model context_20_repeat --context 20 
