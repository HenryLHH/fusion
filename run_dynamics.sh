#!/bin/bash
python scripts/train_dt_dynamics.py --rtg_scale 400.0 --ctg_scale 10.0 \
    --dataset dataset_mixed --model dynamics_mixed --context 20 