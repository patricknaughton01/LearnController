"""
    consensus_test.py
    This module is used to test how similar the predictions of different
    models are. It grabs model files from the paths listed in `model_paths`
    and selects a scene at random from the list of `scenes`. It then tests
    each model on this scene (resetting the scene each time) and records
    the final robot position each model decides on.

    This module creates two files: a text file which lists in a tab separated
    format the scene, xbar, ybar, varx, and vary of the final robot positions.
    It also outputs a .p file which is a pickle file containing the state
    dictionaries of each trial for each model (the verticies of all the
    obstacles, the starting position of all the agents and their original
    goals, and the final position of the robot).
"""


import configparser
import random
import copy
import pickle

from rl_model import Controller
from utils import *
from simulator import Simulator


def main():
    args = parse_args()
    model_paths = [
        "tests/barge_fail_test_3/model.tar",
        "tests/barge_fail_test_4/model.tar",
        "tests/barge_fail_test_5/model.tar",
        "tests/barge_fail_test_6/model.tar",
        "tests/barge_fail_test_7/model.tar",
    ]
    scenes = [
        "barge_in_right",
        "barge_in_left",
        "barge_in_top",
        "barge_in_bottom",
    ]
    state_dicts = []
    timesteps = 300
    epsilon = 0.05
    trials = 10
    out_path = "barge_in_consensus"
    out_file = open(out_path + ".txt", "w")
    out_file.write('scene\txbar\tybar\tvarx\tvary\n')
    pickle_file = open(out_path + ".p", "wb")
    for i in range(trials):
        state_dict = {}
        sim = Simulator(scene=random.choice(scenes))
        orig_agent_poses = []
        orig_goals = copy.deepcopy(sim.goals)
        for agent in sim.agents:
            orig_agent_poses.append(sim.sim.getAgentPosition(agent))
        state_dict["agents"] = orig_agent_poses
        state_dict["goals"] = orig_goals
        state_dict["obstacles"] = sim.obstacles
        final_robot_positions = []
        x_accum = 0
        y_accum = 0
        # Test each model
        for path in model_paths:
            model_config = configparser.RawConfigParser()
            model_config.read(args.model_config)
            model = Controller(model_config,
                               model_type=args.model_type)
            model.load_state_dict(torch.load(path)["state_dict"])
            model.eval()
            h_t = None
            # Run the simulation with this model
            for t in range(timesteps):
                action, h_t = model.select_action(
                    sim.state(), h_t, epsilon=epsilon
                )
                sim.do_step(action)
            # Used to calculate the mean and variance of the final positions
            rpos = sim.sim.getAgentPosition(sim.robot_num)
            final_robot_positions.append(rpos)
            x_accum += rpos[0]
            y_accum += rpos[1]
            # Reset the scene
            for k, pos in enumerate(orig_agent_poses):
                sim.sim.setAgentPosition(k, pos)
            for k, pos in enumerate(orig_goals):
                sim.goals[k] = pos
        # Record what we saw
        state_dict["final_robot_positions"] = final_robot_positions
        xbar = x_accum/len(final_robot_positions)
        ybar = y_accum/len(final_robot_positions)
        state_dict["xbar"] = xbar
        state_dict["ybar"] = ybar
        varx, vary = variance(xbar, ybar, final_robot_positions)
        state_dict["varx"] = varx
        state_dict["vary"] = vary
        state_dicts.append(state_dict)
        out_file.write("{}\t{}\t{}\t{}\t{}\n".format(
            sim.scene, xbar, ybar, varx, vary
        ))
    pickle.dump(state_dicts, pickle_file)
    pickle_file.close()
    out_file.close()


def variance(xbar, ybar, data_points):
    """Calculate the sample variance of data_points given xbar and ybar.
    This function assumes xbar and ybar are accurate.

    :param float xbar: Sample mean of the x's of `data_points`
    :param float ybar: Sample mean of the y's of `data_points`
    :param tuple data_points: tuple of x,y pairs to find the variance of.
    :return: The sample variance of the x coordinates and y coordinates in
        `data_points`.
        :rtype: tuple (varx, vary)
    """
    sumx = 0
    sumy = 0
    for pt in data_points:
        sumx += (xbar - pt[0]) ** 2
        sumy += (ybar - pt[1]) ** 2
    n = len(data_points)
    return sumx / (n - 1), sumy / (n - 1)


if __name__ == "__main__":
    main()
