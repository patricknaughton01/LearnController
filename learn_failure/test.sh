#!/usr/bin/env bash

python test.py \
--max_timesteps 0 --num_episodes 10 --scene dynamic_barge_in_success \
--success_max_ts 10 \
--success_path learn_general_controller/log/sim_barge_in_short/seed_34_bootstrap_False_M_1/model_m_0.tar \
--model_path tests/dynamic_barge/dynamic_barge_model_38/model_initsuccess.tar \
--epsilon=0 \
