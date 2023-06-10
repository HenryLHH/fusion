# !/bin/bash
python ssr/encoder/model_state_est.py --mode train --model state_est_causal \
    --encoder image_spurious --causal True