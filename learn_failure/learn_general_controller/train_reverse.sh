#!/bin/sh

python train.py \
--num_epochs 50 \
--print_every 1 \
--M 1 \
--train_data_name sbi_10000_final_rev \
--test_data_name sbi_10000_final_rev_val \
--dropout 0.5 \
--seed 100 \
--batch_size 64 \
--model_config configs/reverse_model.config \
--reverse \
