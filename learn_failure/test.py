import torch
import learn_general_controller.model
import configparser

from rl_model import Controller
from utils import *
from simulator import Simulator


def main():
    args = parse_args()
    model_path = "dynamic_barge_in/model_soft_update_0.tar"
    scene = args.scene
    timesteps = args.max_timesteps
    epsilon = 0.05
    for i in range(args.num_episodes):
        out_path = args.scene + "_{}.txt".format(i)
        out_file = open(out_path, "w")
        model_config = configparser.RawConfigParser()
        model_config.read(args.model_config)
        model = Controller(model_config,
                           model_type=args.model_type)  # model_type = crossing
        model.load_state_dict(torch.load(model_path)["state_dict"])
        model.eval()
        sim = Simulator(scene=scene, file=out_file)
        success_model = None
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
            except IOError:
                success_model = None
                print("Couldn't open file at {}".format(args.success_path))
        if success_model is not None:
            sim.forward_simulate(success_model, max_ts=args.success_max_ts,
                                 failure_func=lambda p : False)
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
        print("Reward: ", total_reward.item())
        out_file.close()


if __name__ == "__main__":
    main()
