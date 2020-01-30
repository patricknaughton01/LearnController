#!/bin/sh

python train.py \
--num_epochs 100 \
--print_every 1 \
--M 1 \
--train_data_name sim_barge_in_short_safe_final_train \
--test_data_name sim_barge_in_short_safe_final_val \
--dropout 0.3 \
--seed 1 \
--batch_size 128 \
