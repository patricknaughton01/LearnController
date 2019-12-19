#!/bin/sh

python train.py \
--num_epochs 200 \
--print_every 1 \
--M 1 \
--train_data_name simulate_barge_in_heading \
--test_data_name simulate_barge_in_heading \
--dropout 0.5 \
--seed 4567 \
--batch_size 32
