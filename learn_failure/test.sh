#!/usr/bin/env bash

python test.py \
--max_timesteps 0 --num_episodes 5 --scene dynamic_barge_in \
--success_path learn_general_controller/log/simulate_barge_in_0.1_patrick2/seed_1234_bootstrap_False_M_1/model_m_0.tar \
--model_path tests/dynamic_barge/dynamic_barge_model_37/model_fixed_close_cont.tar
