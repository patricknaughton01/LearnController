#!/usr/bin/env bash

python test_failures.py \
--max_timesteps 0 --num_episodes 100 --scene dynamic_barge_in \
--success_max_ts 15 \
--success_path learn_general_controller/log/sim_barge_in_short_safe_final_train/seed_12_bootstrap_False_M_1_reverse_False/model_m_0.tar \
--reverse_path learn_general_controller/log/sbi_small_10000_rev/seed_1_bootstrap_False_M_1_reverse_True/model_m_0.tar \
--model_path tests/success_controller_reverse/initsuccess_final/model_short_initsuccess_final.tar \
--epsilon=0
