#!/bin/sh

python train.py \
--num_epochs 200 \
--print_every 1 \
--M 1 \
--train_data_name sim_barge_in_50 \
--test_data_name sim_barge_in_50 \
--dropout 0.5 \
--seed 3456 \
--batch_size 32 \
--model_path log/sim_barge_in_50/seed_2345_bootstrap_False_M_1/model_m_0.tar
