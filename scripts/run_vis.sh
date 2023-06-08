#!/bin/bash
for cost in 5.0 7.5 10.0 12.5 15.0 17.5 20.0
do
    python tools/1_eval_dt.py --method ssr --model context_20_single_dynamics_value --context 20 \
        --horizon 1000 --cost $cost --single_env --dynamics --value
done
# python train/train_dt_single_env.py --rtg_scale 400.0 --ctg_scale 10.0 \
#     --dataset data_mixed_dynamics --model context_20_repeat --context 20 
