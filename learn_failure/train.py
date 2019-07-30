import numpy as np
import torch
import torch.nn as nn
import os
import learn_general_controller.model
from utils import *
import rl_model
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

    scenes = [
        "dynamic_barge_in",
        #"dynamic_barge_in_left",
        #"dynamic_barge_in_top",
        #"dynamic_barge_in_bottom",
        "barge_in",
        #"barge_in_left",
        #"barge_in_top",
        #"barge_in_bottom",
    ]

    try:
        success_model_config = configparser.RawConfigParser()
        success_model_config.read("learn_general_controller/configs/model.config")
        success_model = learn_general_controller.model.Controller(
            success_model_config,model_type=args.model_type
        )
        success_model.load_state_dict(
            torch.load(args.success_path)["state_dict"])
        success_model.eval()
    except IOError:
        success_model = None
        print("Couldn't open file {}".format(args.success_path))

    model_config = configparser.RawConfigParser()
    model_config.read(args.model_config)
    model = rl_model.Controller(model_config, model_type=args.model_type)
    # This allows us to suspend training and start with the model that gets
    # saved instead of having to start over every time
    if args.model_path != "":
        try:
            model.load_state_dict(torch.load(args.model_path)["state_dict"])
        except IOError:
            print("Couldn't open file {}".format(args.model_path))
    model.train()
    print(model)
    trainer = Trainer(model, config, success_model=success_model)
    path = log_path
    file_name = "model_" + args.name + ".tar"
    if not os.path.exists(path):
        os.makedirs(path)
    try:
        trainer.run(scenes=scenes)
        torch.save({'state_dict': model.state_dict()},
                   file_name)
    except KeyboardInterrupt:
        torch.save({'state_dict': model.state_dict()},
                   file_name)


if __name__ == "__main__":
    main()
