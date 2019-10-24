#!/usr/bin/env bash

python train.py \
--max_timesteps 100 --num_episodes 1000000 --name initsuccess \
--success_path learn_general_controller/log/simulate_barge_in_0.1_patrick2/seed_1234_bootstrap_False_M_1/model_m_0.tar
