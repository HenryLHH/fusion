#!/bin/bash

# python tools/6_eval_checkpoint.py --method gsa --model gsa_single --context 20 \
#         --horizon 1000 --cost 5.0 --single_env --ep_num 50 # --generalize
# python tools/6_eval_checkpoint.py --method ssr_nc --model context_20_nocost_single --context 20 \
#         --horizon 1000 --reward 20.0 --cost 40.0 --single_env --ep_num 50 # --generalize
# python tools/6_eval_checkpoint.py --method ssr --model context_20_single --context 20 \
#         --horizon 1000 --cost 20.0 --single_env --ep_num 50 # --generalize
# python tools/6_eval_checkpoint.py --method ssr --model context_20_single_dynamics_value --context 20 \
#         --horizon 1000 --cost 5.0  --dynamics --value --single_env --ep_num 1 --num_envs 1 # --generalize # --state_pred # --generalize
# python tools/6_eval_checkpoint.py --method ssr --model context_20_single_dynamics_value --context 20 \
#         --horizon 1000 --cost 5.0  --dynamics --value --single_env --ep_num 1 --num_envs 1 --num_trials 1 # --state_pred # --generalize
# python tools/6_eval_checkpoint.py --method ssr --model context_20_single_dynamics_value --context 20 \
#         --horizon 1000 --cost 80.0  --dynamics --value --single_env --ep_num 50 --state_pred --causal # --generalize
# python tools/6_eval_checkpoint.py --method ssr --model context_20_single --context 20 \
#         --horizon 1000 --cost 5.0  --single_env --ep_num 50 # --state_pred --causal #  --generalize
# python tools/6_eval_checkpoint.py --method ssr --model context_20_single --context 20 \
#         --horizon 1000 --cost 30.0 --single_env --ep_num 16 --state_pred --causal #  --generalize

# python tools/6_eval_checkpoint.py --method ssr --model context_20_single --context 20 \
#         --horizon 1000 --cost 20.0 --single_env --ep_num 16 # --state_pred --causal #  --generalize
# python tools/6_eval_checkpoint.py --method ssr --model context_20_single --context 20 \
#         --horizon 1000 --cost 0.0 --single_env --ep_num 16 --state_pred --causal #  --generalize

# python tools/6_eval_checkpoint.py --method icil --model icil_state_single \
#         --horizon 1000 --cost 5.0 --dynamics --value --single_env # --state_pred --causal # --generalize # --causal # --generalize
# python tools/6_eval_checkpoint.py --method bnn --model bnn_single \
#         --horizon 1000 --cost 5.0 --dynamics --value --single_env # --generalize
# python tools/6_eval_checkpoint.py --method bc --model bc_state_single_ \
#         --horizon 1000 --cost 5.0 --dynamics --value --single_env # --generalize 
# python tools/6_eval_checkpoint.py --method bc --model bc_state_single_ \
#         --horizon 1000 --cost 5.0 --dynamics --value --single_env --state_pred --causal --generalize 
# python tools/6_eval_checkpoint.py --method bcql  \
#         --model "logs/MetaDrive-TopDown-v0-cost-10/BCQL_cost_scale10.0_num_workers16_reward_scale40.0-a5f8/BCQL_cost_scale10.0_num_workers16_reward_scale40.0-a5f8/BCQL_cost_scale10.0_num_workers16_reward_scale40.0-a5f8/checkpoint/model_best.pt" \
#         --horizon 1000 --cost 5.0 --dynamics --value --single_env  --ep_num 50 --generalize
# python tools/6_eval_checkpoint.py --method bcql  \
#         --model "logs/MetaDrive-TopDown-v0-cost-1/BCQL_cost1_datasetdataset_mixed_single_post_single_envTrue-f05d/BCQL_cost1_datasetdataset_mixed_single_post_single_envTrue-f05d/checkpoint/model_best.pt" \
#         --horizon 1000 --cost 5.0 --dynamics --value --single_env  --ep_num 50  --generalize

# python tools/6_eval_checkpoint.py --method bearl  \
#         --model "logs/MetaDrive-TopDown-v0-cost-10/BEARL_datasetdataset_mixed_single_post_eval_every2500_single_envTrue-7843/BEARL_datasetdataset_mixed_single_post_eval_every2500_single_envTrue-7843/checkpoint/model_best.pt" \
#         --horizon 1000 --cost 5.0 --dynamics --value --single_env  --ep_num 50

# python tools/6_eval_checkpoint.py --method bearl  \
#         --model "logs/MetaDrive-TopDown-v0-cost-10/BEARL_datasetdataset_mixed_single_post_eval_every2500_single_envTrue-7843/BEARL_datasetdataset_mixed_single_post_eval_every2500_single_envTrue-7843/checkpoint/model_best.pt" \
#         --horizon 1000 --cost 5.0 --dynamics --value --single_env --generalize  --ep_num 50

# python tools/6_eval_checkpoint.py --method bearl  \
#         --model "logs/MetaDrive-TopDown-v0-cost-1/BEARL_cost1_datasetdataset_mixed_single_post_eval_every2500_single_envTrue-ab87/BEARL_cost1_datasetdataset_mixed_single_post_eval_every2500_single_envTrue-ab87/checkpoint/model.pt" \
#         --horizon 1000 --cost 5.0 --dynamics --value --single_env --ep_num 50 # --generalize

# python tools/6_eval_checkpoint.py --method cpq  \
#         --model "logs/MetaDrive-TopDown-v0-cost-1/CPQ_cost1_datasetdataset_mixed_single_post_single_envTrue-2432/CPQ_cost1_datasetdataset_mixed_single_post_single_envTrue-2432/checkpoint/model_best.pt" \
#         --horizon 1000 --cost 5.0 --dynamics --value --single_env --ep_num 50 # --generalize

python tools/6_eval_checkpoint.py --method ssr --model context_20_single_safe_dynamics_value --context 20 \
        --horizon 1000 --cost 5.0  --dynamics --value --single_env --ep_num 50