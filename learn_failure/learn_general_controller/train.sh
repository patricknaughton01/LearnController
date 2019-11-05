#!/bin/sh

python train.py \
--num_epochs 20 \
--print_every 10 \
--M 1 \
--train_data_name simulate_barge_in_0.1_patrick2 \
--test_data_name simulate_barge_in_0.1_patrick2 \
--dropout 0.01 \
--seed 1234567890
