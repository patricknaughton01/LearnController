#!/usr/bin/env bash
python test.py \
--num_epochs 10 \
--print_every 1 \
--M 1 \
--train_data_name simulate_barge_in_heading \
--test_data_name simulate_barge_in_heading \
--seed 3456
