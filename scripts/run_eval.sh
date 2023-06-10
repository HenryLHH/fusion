#!/bin/bash

# python tools/6_eval_checkpoint.py --method ssr --model context_20_single --context 20 \
#         --horizon 1000 --cost 5.0 --single_env --ep_num 50 # --generalize
# python tools/6_eval_checkpoint.py --method ssr --model context_20_single_dynamics_value --context 20 \
#         --horizon 1000 --cost 5.0  --dynamics --value --single_env --ep_num 50 --generalize
# python tools/6_eval_checkpoint.py --method icil --model icil_state_single \
#         --horizon 1000 --cost 5.0 --dynamics --value --single_env --generalize
# python tools/6_eval_checkpoint.py --method bnn --model bnn_single_full \
#         --horizon 1000 --cost 5.0 --dynamics --value --generalize --single_env
# python tools/6_eval_checkpoint.py --method bc --model bc_state_single_ \
#         --horizon 1000 --cost 5.0 --dynamics --value --single_env # --generalize 
# python tools/6_eval_checkpoint.py --method bc --model bc_state_single_ \
#         --horizon 1000 --cost 5.0 --dynamics --value --single_env --generalize 
# python train/train_dt_single_env.py --rtg_scale 400.0 --ctg_scale 10.0 \
#     --dataset data_mixed_dynamics --model context_20_repeat --context 20 

python tools/6_eval_checkpoint.py --method bcql  \
        --model "logs/MetaDrive-TopDown-v0-cost-10/BCQL_cost_scale10.0_num_workers16_reward_scale40.0-a5f8/BCQL_cost_scale10.0_num_workers16_reward_scale40.0-a5f8/BCQL_cost_scale10.0_num_workers16_reward_scale40.0-a5f8/checkpoint/model_best.pt" \
        --horizon 1000 --cost 5.0 --dynamics --value --single_env  --ep_num 50 --generalize
# python tools/6_eval_checkpoint.py --method bearl  \
#         --model "logs/MetaDrive-TopDown-v0-cost-10/BEARL_datasetdataset_mixed_single_post_eval_every2500_single_envTrue-7843/BEARL_datasetdataset_mixed_single_post_eval_every2500_single_envTrue-7843/checkpoint/model_best.pt" \
#         --horizon 1000 --cost 5.0 --dynamics --value --single_env  --ep_num 50

# python tools/6_eval_checkpoint.py --method bearl  \
#         --model "logs/MetaDrive-TopDown-v0-cost-10/BEARL_datasetdataset_mixed_single_post_eval_every2500_single_envTrue-7843/BEARL_datasetdataset_mixed_single_post_eval_every2500_single_envTrue-7843/checkpoint/model_best.pt" \
#         --horizon 1000 --cost 5.0 --dynamics --value --single_env --generalize  --ep_num 50
