"""generate_training_data.py
This script uses a trained model to generate training data to predict final
states from the initial state the robot observes.

"""
import utils
import torch
import configparser
import pickle
import random

from rl_model import Controller
from simulator import Simulator


def main():
    args = utils.parse_args()
    model_path = "tests/dynamic_barge/dynamic_barge_in/model.tar"
    scenes = [
        "dynamic_barge_in",
        "dynamic_barge_in_left",
        "dynamic_barge_in_top",
        "dynamic_barge_in_bottom",
        "barge_in",
        "barge_in_left",
        "barge_in_top",
        "barge_in_bottom",
    ]
    timesteps = args.max_timesteps
    epsilon = args.epsilon
    model_config = configparser.RawConfigParser()
    model_config.read(args.model_config)
    model = Controller(model_config,
                       model_type=args.model_type)  # model_type = crossing
    model.load_state_dict(torch.load(model_path)["state_dict"])
    model.eval()
    out_path = args.scene + "_final_states.p"
    out_file = open(out_path, "wb")
    final_map = []
    for i in range(args.num_episodes):
        try:
            print("{}/{}".format(i, args.num_episodes), end="\r")
            sim = Simulator(scene=random.choice(scenes))
            init_state = sim.state()[sim.robot_num]
            h_t = None
            for t in range(timesteps):
                action, h_t = model.select_action(
                    sim.state(), h_t, epsilon=epsilon
                )
                sim.do_step(action)
            # Map the initial state to the final position of the robot - this
            # will serve as training data for another network.
            final_map.append(
                (init_state, sim.sim.getAgentPosition(sim.robot_num))
            )
        except KeyboardInterrupt:
            break
    pickle.dump(final_map, out_file)
    out_file.close()


if __name__ == "__main__":
    main()
