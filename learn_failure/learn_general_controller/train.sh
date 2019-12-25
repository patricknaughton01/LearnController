#!/bin/sh

python train.py \
--num_epochs 5000 \
--print_every 1 \
--M 1 \
--train_data_name sim_barge_in_heading_less_data \
--test_data_name sim_barge_in_heading_less_data \
--dropout 0.5 \
--seed 1234 \
--batch_size 64
