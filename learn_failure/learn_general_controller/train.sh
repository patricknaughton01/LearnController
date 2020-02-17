#!/bin/sh

python train.py \
--num_epochs 500 \
--print_every 1 \
--M 1 \
--train_data_name sim_barge_in_short_safe_final_train \
--test_data_name sim_barge_in_short_safe_final_val \
--dropout 0.5 \
--seed 16 \
--batch_size 128 \
--model_path log/sim_barge_in_short_safe_final_train/seed_15_bootstrap_False_M_1_reverse_False/model_m_0.tar
