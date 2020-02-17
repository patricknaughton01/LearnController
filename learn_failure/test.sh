#!/usr/bin/env bash

python test.py \
--max_timesteps 0 --num_episodes 1 --scene dynamic_barge_in_success \
--success_max_ts 15 \
--success_path learn_general_controller/log/sim_barge_in_short_safe_final_train/seed_15_bootstrap_False_M_1_reverse_False/model_m_0.tar \
--reverse_path learn_general_controller/log/sbi_10000_final_rev/seed_100_bootstrap_False_M_1_reverse_True/model_m_0.tar \
--model_path tests/final/test_2/model_short_initsuccess_final_2.tar \
--epsilon=0 --record --conf --conf_val 0.95
