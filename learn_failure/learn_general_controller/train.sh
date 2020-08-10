#!/bin/sh

python3 train.py \
--num_epochs 500 \
--print_every 1 \
--M 1 \
--train_data_name biwi_ped \
--test_data_name biwi_ped \
--dropout 0.5 \
--seed 16 \
--batch_size 10
