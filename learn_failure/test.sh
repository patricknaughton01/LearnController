#!/usr/bin/env bash

python test.py \
--max_timesteps 5 --num_episodes 4 --scene dynamic_barge_in \
--success_max_ts 10 \
--success_path learn_general_controller/log/sim_barge_in_short_safe_2/seed_3_bootstrap_False_M_1_reverse_False/model_m_0.tar \
--reverse_path learn_general_controller/log/sim_barge_in_short_safe_2_train3_single/seed_2_bootstrap_False_M_1_reverse_True/model_m_0.tar \
--model_path tests/success_controller_reverse/initsuccess/model_short_initsuccess.tar \
--epsilon=0
