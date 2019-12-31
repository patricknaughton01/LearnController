#!/bin/sh

python train.py \
--num_epochs 600 \
--print_every 1 \
--M 1 \
--train_data_name sim_barge_in_short_safe \
--test_data_name sim_barge_in_short_safe \
--dropout 0.5 \
--seed 234 \
--batch_size 32 \
--reverse \
--model_path log/sim_barge_in_short_safe/seed_1234_bootstrap_False_M_1_reverse_False/model_m_0.tar
