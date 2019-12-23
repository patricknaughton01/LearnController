#!/bin/sh

python train.py \
--num_epochs 100000 \
--print_every 1 \
--M 1 \
--train_data_name sim_barge_in_1 \
--test_data_name sim_barge_in_1 \
--dropout 0.5 \
--seed 2345 \
--batch_size 1 \
--model_path log/sim_barge_in_1/seed_234_bootstrap_False_M_1/model_m_0.tar
