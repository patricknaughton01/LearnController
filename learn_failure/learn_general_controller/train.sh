#!/bin/sh

python train.py \
--num_epochs 200 \
--print_every 1 \
--M 1 \
--train_data_name sim_barge_in_short \
--test_data_name sim_barge_in_short \
--dropout 0.5 \
--seed 4567 \
--batch_size 32 \
--model_path log/sim_barge_in_short/seed_4_bootstrap_False_M_1/model_m_0.tar \
--reverse
