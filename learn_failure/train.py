import numpy as np
import torch
import torch.nn as nn
import os
from utils import *
from rl_model import *
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
    log_path = os.path.dirname(os.path.realpath(__file__))
    os.path.join(log_path, "log", args.model_type)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    config['log_path'] = log_path

    model_config = configparser.RawConfigParser()
    model_config.read(args.model_config)
    model = Controller(model_config, model_type=args.model_type) # model_type = crossing
    model.train()
    print(model)
    trainer = Trainer(model, config)
    try:
        trainer.run(args.scene)
        torch.save({'state_dict': model.state_dict()},
                   log_path + '/model' + '.tar')
    except KeyboardInterrupt:
        torch.save({'state_dict': model.state_dict()},
                   log_path + '/model' + '.tar')


if __name__ == "__main__":
    main()
