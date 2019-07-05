import numpy as np
import torch 
import torch.nn as nn
import os 
from utils import *
from model import *
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

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    config['log_path'] = log_path

    train_data_path = args.data_path + '/' + args.train_data_name
    train_states = np.load(train_data_path + '/states.npy', allow_pickle=True)
    # check pytorch note for the description of what the states.npy contains
    # print('train_states')
    # print(train_states.shape)
    # print(train_states)
    train_seq_lengths = np.load(train_data_path + '/seq_lengths.npy', allow_pickle=True)
    # print('train_seq_lengths')
    # print(train_seq_lengths.shape)
    train_num_humans = np.load(train_data_path + '/num_humans.npy', allow_pickle=True)
    # print('train_num_humans')
    # print(train_num_humans.shape)
    # print('Total train data size: %d' % len(train_seq_lengths))
    train_loader = lambda: dataloader((train_states, train_seq_lengths, train_num_humans), config, shuffle=True)

    test_data_path = args.data_path + '/' + args.test_data_name
    test_states = np.load(test_data_path + '/states.npy', allow_pickle=True)
    test_seq_lengths = np.load(test_data_path + '/seq_lengths.npy', allow_pickle=True)
    test_num_humans = np.load(test_data_path + '/num_humans.npy', allow_pickle=True)
    print('Total test data size: %d' % len(test_seq_lengths))
    val_loader = lambda: dataloader((test_states, test_seq_lengths, test_num_humans), config, shuffle=False)

    model_config = configparser.RawConfigParser()
    model_config.read(args.model_config)
    # print('model_config')
    # print(args.model_config)
    for m in range(args.M):
        print('start training model %d ...' % m)
        model = Controller(model_config, model_type=args.model_type) # model_type = crossing
        print(model)
        trainer = Trainer(model, train_loader, val_loader, config)
        trainer.run()
        torch.save({'state_dict': model.state_dict()}, log_path + '/model_m_' + str(m) + '.tar')
        print('finished one model')

if __name__ == "__main__":
    main()
