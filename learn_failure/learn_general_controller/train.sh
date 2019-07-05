#!/bin/sh

python train.py \
--seed 1234 \
--num_epochs 60 \
--print_every 10 \
--M 5 \
--train_data_name simulate_barge_in \
--test_data_name simulate_barge_in