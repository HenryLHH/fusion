#!/bin/bash
# for cost in 18.0 20.0 22.0 24.0 26.0
# do
#     python tools/6_eval_checkpoint.py --method ssr --model context_20_single --context 20 \
#             --horizon 1000 --cost $cost --single_env --ep_num 50 # --generalize
# done
# for reward in 0 50 100 150 200 250 300 350
# do
#     python tools/6_eval_checkpoint.py --method ssr --model context_20_single_dynamics_value --context 20 \
#             --horizon 1000 --reward $reward --cost 25 --single_env --ep_num 50 --dynamics --value # --generalize
# done


for task in "R" "X" "T" "O" 
do
    python tools/6_eval_checkpoint.py --method ssr --model context_20_dynamics_value --context 20 \
            --horizon 1000 --single_env --ep_num 50 --cost 5 --dynamics --value --task $task # --generalize
done

# for task in "S" "C" "R" "X" "T" "O" 
# do
#     python tools/6_eval_checkpoint.py --method expert --horizon 1000 \
#         --single_env --ep_num 50 --task $task # --generalize
# done

# python train/train_dt_single_env.py --rtg_scale 400.0 --ctg_scale 10.0 \
#     --dataset data_mixed_dynamics --model context_20_repeat --context 20 

# 0.82 0.17 0.67 0.86 0.79 0.40
# 0.07 0.02 0.06 0.05 0.03 0.04
# straight
# eval avg oor: 0.00\scriptsize{$\pm$0.00}                                                                                                                                                                      
# eval avg overtime: 0.00\scriptsize{$\pm$0.00}                                                                                                                                                                 
# eval avg cost: 0.43\scriptsize{$\pm$0.02}  
# curve (ood)
# eval avg reward: 101.31\scriptsize{$\pm$10.43}
# eval avg succ: 0.18\scriptsize{$\pm$0.04}
# eval avg crash: 0.43\scriptsize{$\pm$0.08}
# eval avg oor: 0.38\scriptsize{$\pm$0.07}
# eval avg overtime: 0.01\scriptsize{$\pm$0.02}
# eval avg cost: 1.12\scriptsize{$\pm$0.12}

# ramp
# eval avg reward: 120.48\scriptsize{$\pm$2.64}
# eval avg succ: 0.63\scriptsize{$\pm$0.02}
# eval avg crash: 0.37\scriptsize{$\pm$0.01}
# eval avg oor: 0.00\scriptsize{$\pm$0.01}
# eval avg overtime: 0.00\scriptsize{$\pm$0.00}
# eval avg cost: 1.50\scriptsize{$\pm$0.16}

# X
# eval avg reward: 131.81\scriptsize{$\pm$2.93}
# eval avg succ: 0.87\scriptsize{$\pm$0.02}
# eval avg crash: 0.09\scriptsize{$\pm$0.04}
# eval avg oor: 0.04\scriptsize{$\pm$0.03}
# eval avg overtime: 0.00\scriptsize{$\pm$0.00}
# eval avg cost: 0.97\scriptsize{$\pm$0.15}

# T
# eval avg reward: 135.10\scriptsize{$\pm$3.70}
# eval avg succ: 0.78\scriptsize{$\pm$0.03}
# eval avg crash: 0.04\scriptsize{$\pm$0.01}
# eval avg oor: 0.19\scriptsize{$\pm$0.03}
# eval avg overtime: 0.00\scriptsize{$\pm$0.00}
# eval avg cost: 1.46\scriptsize{$\pm$0.16}

# O

# eval avg reward: 131.01\scriptsize{$\pm$6.12}
# eval avg succ: 0.38\scriptsize{$\pm$0.09}
# eval avg crash: 0.50\scriptsize{$\pm$0.06}
# eval avg oor: 0.12\scriptsize{$\pm$0.04}
# eval avg overtime: 0.00\scriptsize{$\pm$0.00}
# eval avg cost: 0.81\scriptsize{$\pm$0.06}


# expert

# straight
# eval avg reward: 99.38\scriptsize{$\pm$34.03}
# eval avg succ: 0.80\scriptsize{$\pm$0.33}
# eval avg crash: 0.19\scriptsize{$\pm$0.32}
# eval avg oor: 0.00\scriptsize{$\pm$0.00}
# eval avg overtime: 0.01\scriptsize{$\pm$0.02}
# eval avg cost: 12.03\scriptsize{$\pm$9.64}

# curve
# eval avg reward: 96.01\scriptsize{$\pm$39.80}
# eval avg succ: 0.29\scriptsize{$\pm$0.20}
# eval avg crash: 0.52\scriptsize{$\pm$0.18}
# eval avg oor: 0.00\scriptsize{$\pm$0.00}
# eval avg overtime: 0.20\scriptsize{$\pm$0.05}
# eval avg cost: 0.52\scriptsize{$\pm$0.18}

# eval avg reward: 161.23\scriptsize{$\pm$74.36}
# eval avg succ: 0.74\scriptsize{$\pm$0.37}
# eval avg crash: 0.26\scriptsize{$\pm$0.37}
# eval avg oor: 0.00\scriptsize{$\pm$0.00}
# eval avg overtime: 0.00\scriptsize{$\pm$0.00}
# eval avg cost: 19.98\scriptsize{$\pm$15.94}

# ramp
# method: expert, checkpoint: encoder
# ============================================================
# eval avg reward: 120.51\scriptsize{$\pm$53.76}
# eval avg succ: 0.75\scriptsize{$\pm$0.37}
# eval avg crash: 0.25\scriptsize{$\pm$0.37}
# eval avg oor: 0.00\scriptsize{$\pm$0.00}
# eval avg overtime: 0.00\scriptsize{$\pm$0.00}
# eval avg cost: 13.62\scriptsize{$\pm$10.96}

# X
# eval avg reward: 115.57\scriptsize{$\pm$7.00}
# eval avg succ: 0.86\scriptsize{$\pm$0.12}
# eval avg crash: 0.14\scriptsize{$\pm$0.12}
# eval avg oor: 0.00\scriptsize{$\pm$0.00}
# eval avg overtime: 0.00\scriptsize{$\pm$0.00}
# eval avg cost: 1.88\scriptsize{$\pm$2.42}

# eval avg reward: 87.22\scriptsize{$\pm$13.78}
# eval avg succ: 0.44\scriptsize{$\pm$0.22}
# eval avg crash: 0.56\scriptsize{$\pm$0.21}
# eval avg oor: 0.01\scriptsize{$\pm$0.02}
# eval avg overtime: 0.00\scriptsize{$\pm$0.00}
# eval avg cost: 9.83\scriptsize{$\pm$7.83}

# T
# eval avg reward: 95.48\scriptsize{$\pm$14.13}
# eval avg succ: 0.56\scriptsize{$\pm$0.22}
# eval avg crash: 0.39\scriptsize{$\pm$0.19}
# eval avg oor: 0.03\scriptsize{$\pm$0.03}
# eval avg overtime: 0.02\scriptsize{$\pm$0.04}
# eval avg cost: 10.31\scriptsize{$\pm$8.46}

# O
# eval avg reward: 132.57\scriptsize{$\pm$10.67}
# eval avg succ: 0.48\scriptsize{$\pm$0.11}
# eval avg crash: 0.50\scriptsize{$\pm$0.11}
# eval avg oor: 0.02\scriptsize{$\pm$0.01}
# eval avg overtime: 0.00\scriptsize{$\pm$0.00}
# eval avg cost: 2.63\scriptsize{$\pm$2.94}

# eval avg reward: 97.27\scriptsize{$\pm$20.39}
# eval avg succ: 0.26\scriptsize{$\pm$0.16}
# eval avg crash: 0.53\scriptsize{$\pm$0.17}
# eval avg oor: 0.18\scriptsize{$\pm$0.20}
# eval avg overtime: 0.04\scriptsize{$\pm$0.08}
# eval avg cost: 10.80\scriptsize{$\pm$8.79}