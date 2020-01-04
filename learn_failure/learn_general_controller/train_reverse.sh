#!/bin/sh

python train.py \
--num_epochs 100 \
--print_every 1 \
--M 1 \
--train_data_name sim_barge_in_short_safe_2 \
--test_data_name sim_barge_in_short_safe_2 \
--dropout 0.3 \
--seed 2 \
--batch_size 128 \
--reverse
