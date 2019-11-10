#!/bin/sh

python train.py \
--num_epochs 300 \
--print_every 10 \
--M 1 \
--train_data_name simulate_barge_in_0.25_patrick_move_goal \
--test_data_name simulate_barge_in_0.25_patrick_move_goal \
--dropout 0.4 \
--seed 257 \
--model_path /home/patricknaughton01/Downloads/LearnControllers/learn_failure/learn_general_controller/log/simulate_barge_in_0.25_patrick_move_goal/seed_25123456_bootstrap_False_M_1/model_m_0.tar
