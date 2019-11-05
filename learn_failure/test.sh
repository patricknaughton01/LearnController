#!/usr/bin/env bash

python test.py \
--max_timesteps 0 --num_episodes 10 --scene dynamic_barge_in_success \
--success_path /home/patricknaughton01/Downloads/LearnControllers/learn_failure/learn_general_controller/log/simulate_barge_in_0.1_patrick2/seed_987654_bootstrap_False_M_1/model_m_0.tar \
--model_path tests/dynamic_barge/dynamic_barge_model_38/model_initsuccess.tar \
--epsilon=0
