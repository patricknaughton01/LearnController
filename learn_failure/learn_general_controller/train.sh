#!/bin/sh

python train.py \
--num_epochs 100 \
--print_every 1 \
--M 1 \
--train_data_name sim_barge_in_short_safe_2 \
--test_data_name sim_barge_in_short_safe_2 \
--dropout 0.3 \
--seed 3 \
--batch_size 128 \
--model_path log/sim_barge_in_short_safe_2/seed_2_bootstrap_False_M_1_reverse_False/model_m_0.tar
