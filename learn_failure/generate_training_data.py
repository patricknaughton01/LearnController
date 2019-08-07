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
    model_path = args.model_path
    if model_path == "":
        print("You must specify a model")
        return
    scenes = [
        "dynamic_barge_in",
        #"dynamic_barge_in_left",
        #"dynamic_barge_in_top",
        #"dynamic_barge_in_bottom",
        #"barge_in",
        #"barge_in_left",
        #"barge_in_top",
        #"barge_in_bottom",
    ]
    timesteps = args.max_timesteps
    epsilon = args.epsilon
    model_config = configparser.RawConfigParser()
    model_config.read(args.model_config)
    model = Controller(model_config,
                       model_type=args.model_type)  # model_type = crossing
    model.load_state_dict(torch.load(model_path)["state_dict"])
    model.eval()
    out_path = args.name + "_states.p"
    out_file = open(out_path, "wb")
    final_map = []
    for i in range(args.num_episodes):
        try:
            print("{}/{}".format(i, args.num_episodes), end="\r")
            sim = Simulator(scene=random.choice(scenes))
            h_t = None
            trajectory = []
            for t in range(timesteps):
                action, h_t = model.select_action(
                    sim.state(), h_t, epsilon=epsilon
                )
                state = sim.state()
                agents = [(sim.sim.getAgentPosition(a)[0],
                                sim.sim.getAgentPosition(a)[1],
                                sim.sim.getAgentRadius(a),
                                sim.headings[a]) for a in sim.agents]
                sim.do_step(action)
                # Map each state to the resulting location
                trajectory.append(
                    (state, sim.sim.getAgentPosition(sim.robot_num),
                     agents, sim.obstacles)
                )
            final_map.append(trajectory)
        except KeyboardInterrupt:
            break
    print()
    pickle.dump(final_map, out_file)
    out_file.close()


if __name__ == "__main__":
    main()
