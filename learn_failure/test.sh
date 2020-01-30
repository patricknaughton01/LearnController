#!/usr/bin/env bash

python test.py \
--max_timesteps 0 --num_episodes 100 --scene dynamic_barge_in_success \
--success_max_ts 10 \
--success_path learn_general_controller/log/old/sim_barge_in_short_safe_2/seed_3_bootstrap_False_M_1_reverse_False/model_m_0.tar \
--reverse_path learn_general_controller/log/old/sim_barge_in_short_safe_2_train3_single/seed_2_bootstrap_False_M_1_reverse_True/model_m_0.tar \
--model_path tests/success_controller_reverse/initsuccess_final/model_short_initsuccess_final.tar \
--epsilon=0
