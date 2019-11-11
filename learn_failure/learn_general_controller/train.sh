#!/bin/sh

python train.py \
--num_epochs 300 \
--print_every 10 \
--M 1 \
--train_data_name simulate_barge_in_0.25_patrick_move_goal \
--test_data_name simulate_barge_in_0.25_patrick_move_goal \
--dropout 0.4 \
--seed 2516
