#!/usr/bin/env bash

python3 test.py \
--max_timesteps 0 --num_episodes 10 --scene overtake_success \
--success_max_ts 15 \
--success_path learn_general_controller/log/biwi_ped/seed_16_bootstrap_False_M_1_reverse_False/model_m_0.tar \
--reverse_path learn_general_controller/log/sbi_10000_final_rev/seed_100_bootstrap_False_M_1_reverse_True/model_m_0.tar \
--model_path tests/final/test_4/model_short_initsuccess_final_4.tar \
--epsilon=0 --record --conf_val 0.99
