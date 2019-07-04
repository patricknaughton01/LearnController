import numpy as np
import torch
import torch.nn as nn
import pickle
import os
from utils import *
from predict_model import *
from trainer import Trainer
from torch import optim
from glob import glob
import configparser


def main():
    args = parse_args()
    print('args')
    print(args)
    config = vars(args)
    print('config')
    print(config)

    if args.seed > 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    log_path = "log/%s/seed_%d_bootstrap_%s_M_%d" % (
        args.train_data_name,
        args.seed,
        str(args.bootstrap),
        args.M)
    data_path = "barge_in_final_states_1.p"

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    config['log_path'] = log_path

    model_config = configparser.RawConfigParser()
    model_config.read(args.model_config)

    f = open(data_path, "rb")
    all_data = pickle.load(f)
    random.shuffle(all_data)
    print("Total dataset size: {}".format(len(all_data)))
    split_ind = int(len(all_data) * (1 - args.test_percent))
    train_data = all_data[:split_ind]
    val_data = all_data[split_ind:]

    # print('model_config')
    # print(args.model_config)
    for m in range(args.M):
        print('start training model %d ...' % m)
        model = Controller(model_config,
                           model_type=args.model_type)  # model_type = crossing
        print(model)
        trainer = Trainer(model, train_data, val_data, config)
        trainer.run()
        torch.save({'state_dict': model.state_dict()},
                   log_path + '/model_m_' + str(m) + '.tar')
        print('finished one model')


if __name__ == "__main__":
    main()
