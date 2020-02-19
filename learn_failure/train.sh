#!/usr/bin/env bash

python train.py \
--max_timesteps 15 --num_episodes 500 --name short_initsuccess_final_4 \
--success_max_ts 15 \
--success_path learn_general_controller/log/sim_barge_in_short_safe_final_train/seed_16_bootstrap_False_M_1_reverse_False/model_m_0.tar \
--reverse_path learn_general_controller/log/sbi_10000_final_rev/seed_100_bootstrap_False_M_1_reverse_True/model_m_0.tar
