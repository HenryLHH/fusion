# !/bin/bash
python train/train_fusion.py --rtg_scale 300.0 --ctg_scale 40.0 \
    --dataset data_default --model fusion --context 20 \
    --num_workers 4 --num_envs 1 --single_env --dynamics --value

python train/train_icil.py --mode train --model icil_state_small_portion \
        --dataset dataset_mixed_post

python train/train_gsa.py --mode train --model gsa_single_safe \
        --dataset dataset_mixed_post --single_env
    
python train/train_bnn.py --mode train --model bnn_single_full \
        --dataset dataset_mixed_post --single_env

python train/train_bc.py --mode train --model bc_state_safe \
        --dataset dataset_mixed_post --safe --single_env

python train/train_bcql.py --eval_every 2500 --cost_limit 1 \
        --dataset dataset_mixed_post --single_env True

python train/train_bearl.py --eval_every 2500 --cost_limit 1 \
        --dataset dataset_mixed_post --single_env True

