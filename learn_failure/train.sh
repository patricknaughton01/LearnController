#!/usr/bin/env bash

python train.py \
--max_timesteps 10 --num_episodes 10000 --name short_initsuccess_long \
--success_max_ts 10 \
--success_path learn_general_controller/log/sim_barge_in_short_safe_2/seed_3_bootstrap_False_M_1_reverse_False/model_m_0.tar \
--reverse_path learn_general_controller/log/sim_barge_in_short_safe_2_train3_single/seed_2_bootstrap_False_M_1_reverse_True/model_m_0.tar
