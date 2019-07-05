import torch
import configparser

from rl_model import Controller
from utils import *
from simulator import Simulator


def main():
    args = parse_args()
    model_path = "overtaking/model.tar"
    scene = args.scene
    timesteps = 300
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
