import torch
import learn_general_controller.model
import configparser
import random
import time
import os

from tqdm import tqdm
from rl_model import Controller
from utils import *
from simulator import Simulator


def main():
    args = parse_args()
    prefix = "output/"
    if not os.path.isdir(prefix):
       os.mkdir(prefix)
    with open(prefix + "command.txt", "w") as cfile:
        cfile.write(str(args) + "\n")
    model_path = args.model_path
    if model_path == "":
        print("You must provide a model to test for test.py")
        return
    if args.seed is None:
        args.seed = time.time()
    random.seed(args.seed)
    scene = args.scene
    timesteps = args.max_timesteps
    epsilon = args.epsilon
    reward_file = open(prefix + "rewards_{}.txt".format(args.scene), "w")
    for c in range(10):
        failures = 0
        for i in tqdm(range(args.num_episodes)):
            out_path = prefix + args.scene + "_{}.txt".format(i)
            out_file = open(out_path, "w")
            model_config = configparser.RawConfigParser()
            model_config.read(args.model_config)
            model = Controller(model_config,
                               model_type=args.model_type)  # model_type = crossing
            model.load_state_dict(torch.load(model_path)["state_dict"])
            model.eval()
            sim = Simulator(scene=scene, reverse=args.reverse,
                            single=args.single)
            success_model = None
            reverse_model = None
            if args.success_path != "":
                try:
                    success_model_config = configparser.RawConfigParser()
                    success_model_config.read(
                        "learn_general_controller/configs/model.config")
                    success_model = learn_general_controller.model.Controller(
                        success_model_config, model_type=args.model_type
                    )
                    success_model.load_state_dict(
                        torch.load(args.success_path)["state_dict"])
                    success_model.eval()
                    try:
                        reverse_model_config = configparser.RawConfigParser()
                        reverse_model_config.read(
                            "learn_general_controller/configs/model.config")
                        reverse_model = learn_general_controller.model.Controller(
                            reverse_model_config, model_type=args.model_type
                        )
                        reverse_model.load_state_dict(
                            torch.load(args.reverse_path)["state_dict"]
                        )
                        reverse_model.eval()
                    except Exception as e:
                        print("Tried to open reverse model at {}. Needed if "
                              "using a success model".format(args.reverse_path))
                        print(str(e))
                        exit()
                except IOError:
                    success_model = None
                    print("Couldn't open file at {}".format(args.success_path))
            success_ts = 0
            if success_model is not None:
                success_ts = sim.forward_simulate(success_model, reverse_model,
                    max_ts=args.success_max_ts, key=str(i), samples=2,
                                                  conf_val=0.9+(c/100))
                if success_ts < 0:
                    failures += 1
            h_t = None
            total_reward = torch.zeros((1, 1), dtype=torch.float)
            for t in range(timesteps):
                action, h_t = model.select_action(
                    sim.state(), h_t, epsilon=epsilon
                )
                sim.do_step(action)
                reward, _ = sim.reward()
                # Penalize moving
                if action.item() != 0:
                    reward -= 0.01
                total_reward += reward
            #print("Reward: ", total_reward.item())
            reward_file.write(str(total_reward.item()) + "\n")
            out_file.close()
        #reward_file.close()
        print("Failed {} times out of {} at confidence level {}".format(
            failures, args.num_episodes, 0.9+c/100))


if __name__ == "__main__":
    main()
