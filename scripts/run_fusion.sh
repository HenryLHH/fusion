#!/bin/bash
# python train/train_fusion.py --rtg_scale 300.0 --ctg_scale 40.0 \
#     --dataset data_default --model context_20_single_safe_dynamics_value_repeat --context 20 \
#     --num_workers 4 --num_envs 1 --single_env --dynamics --value --use_pretrained --checkpoint context_20_single
# python train/train_dt_single_env.py --rtg_scale 400.0 --ctg_scale 10.0 \
#     --dataset data_default --model dataset_visualize_attn --context 20 --use_pretrained --dynamics --value

# python train/train_fusion.py --rtg_scale 300.0 --ctg_scale 10.0 \
#     --dataset data_mix --model context_20_single_safe_dynamics_value --context 20 \
#     --num_workers 16 --num_envs 16 --single_env --dynamics --value

python train/train_fusion.py --rtg_scale 300.0 --ctg_scale 10.0 \
    --dataset data_mix --model debug_bisim --context 20 \
    --num_envs 16 --num_workers 16 --dynamics --value --single_env --bisim
