#!/bin/sh

python train.py \
--num_epochs 200 \
--print_every 1 \
--M 1 \
--train_data_name sim_barge_in_extra_end_ts \
--test_data_name sim_barge_in_extra_end_ts \
--dropout 0.5 \
--seed 2345 \
--batch_size 32
