#!/bin/sh

python train.py \
--num_epochs 200 \
--print_every 1 \
--M 1 \
--train_data_name sim_barge_in_short_safe_2_train3_single \
--test_data_name sim_barge_in_short_safe_2_train3_single \
--dropout 0.5 \
--seed 2 \
--batch_size 128 \
--reverse
