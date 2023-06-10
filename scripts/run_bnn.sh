#!/bin/bash
python train/train_bnn.py --mode train --model bnn_single_full \
        --dataset dataset_mixed_single_post --single_env
# python train/train_icil.py --mode train --model icil_state_single_repeat_small_portion \
#         --dataset dataset_mixed_single_post --single_env
# python train/train_dt_single_env.py --rtg_scale 400.0 --ctg_scale 10.0 \
#     --dataset data_mixed_dynamics --model context_20_repeat --context 20 
