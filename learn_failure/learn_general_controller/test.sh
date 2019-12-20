#!/usr/bin/env bash
python test.py \
--num_epochs 10 \
--print_every 1 \
--M 1 \
--train_data_name simulate_dynamic_barge_in_test \
--test_data_name simulate_dynamic_barge_in_test \
--seed 3456
