#!/bin/sh

python train.py \
--num_epochs 600 \
--print_every 1 \
--M 1 \
--train_data_name sim_barge_in_short \
--test_data_name sim_barge_in_short \
--dropout 0.5 \
--seed 4 \
--batch_size 32 \
--reverse
