#!/bin/sh

python train.py \
--num_epochs 1 \
--print_every 1 \
--M 1 \
--train_data_name sbi_small_10000_rev \
--test_data_name sbi_small_10000_rev_val \
--dropout 0.5 \
--seed 2 \
--batch_size 128 \
--reverse
