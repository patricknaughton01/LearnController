#!/bin/sh

python train.py \
--num_epochs 200 \
--print_every 1 \
--M 1 \
--train_data_name sim_barge_in_50 \
--test_data_name sim_barge_in_50 \
--dropout 0.5 \
--seed 1234 \
--batch_size 32
