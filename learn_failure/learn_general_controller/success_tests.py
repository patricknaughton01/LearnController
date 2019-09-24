
import torch
import os
import utils
import configparser
import math
import numpy as np
import argparse
from model import Controller
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points


def main():
    args = utils.parse_args()
    train_states = np.load(args.data_path + '/states.npy',
                           allow_pickle=True)
    train_seq_lengths = np.load(args.data_path + '/seq_lengths.npy',
                                allow_pickle=True)
    train_num_humans = np.load(args.data_path + '/num_humans.npy',
                               allow_pickle=True)
    loader = lambda: utils.dataloader((train_states, train_seq_lengths,
                                       train_num_humans), config)
    config = vars(args)
    log_path = os.path.dirname(os.path.realpath(__file__))
    os.path.join(log_path, "log", args.model_type)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    config['log_path'] = log_path
    model_config = configparser.RawConfigParser()
    model_config.read("configs/model.config")
    model = Controller(model_config, model_type=args.model_type)
    try:
        model.load_state_dict(torch.load(args.model_path)["state_dict"])
    except IOError:
        print("Couldn't open file at {}".format(args.model_path))
        return
    with torch.no_grad():
        for states, seq_lengths, targets, future_states in loader():
            x = []
            y = []
            sx = []
            sy = []
            rho = []
            for i in range(100):
                #print(states)
                cur_state = states[0][:1]
                cur_rotated = utils.transform_and_rotate(cur_state)
                om = utils.build_occupancy_maps(utils.build_humans(
                    cur_state[0]))
                state_t = torch.cat([cur_rotated, om.unsqueeze(0)], dim=-1)
                pred_t, h_t = model(state_t, None)
                params = utils.get_coefs(pred_t.unsqueeze(0))
                x.append(params[0].item())
                y.append(params[1].item())
                sx.append(params[2].item())
                sy.append(params[3].item())
                rho.append(params[4].item())
                #print(params)
            print("Avg x: ", avg(x), "\tStd dev x: ", std_dev(x))
            print("Avg y: ", avg(y), "\tStd dev y: ", std_dev(y))
            #print("Covariance: ", cov(x, y))
            print("Correlation: ", cov(x, y) / (std_dev(x) * std_dev(y)))

            print()
            print("Average predicted sx: ", avg(sx), "\tstddev sx: ",
                  std_dev(sx))
            print("Average predicted sy: ", avg(sy), "\tstddev sy: ",
                  std_dev(sy))
            print("Average predicted rho: ", avg(rho), "\tstddev rho: ",
                  std_dev(rho))
            break


def succ_state(sim):
    rpos = sim.sim.getAgentPosition(sim.robot_num)
    rvel = sim.sim.getAgentVelocity(sim.robot_num)
    rrad = sim.sim.getAgentRadius(sim.robot_num)
    v_pref = sim.sim.getAgentMaxSpeed(sim.robot_num)
    theta = math.atan2(rvel[1], rvel[0])
    # Robot's state entry.
    state = [
        rpos[0], rpos[1], rvel[0], rvel[1], rrad,
        sim.overall_robot_goal[0], sim.overall_robot_goal[1], v_pref, theta
    ]
    for agent in sim.agents:
        if agent != sim.robot_num:  # We already accounted for the robot
            pos = sim.sim.getAgentPosition(agent)
            vel = sim.sim.getAgentVelocity(agent)
            rad = sim.sim.getAgentRadius(agent)
            state.extend([pos[0], pos[1], vel[0], vel[1], rad])
    for obs in sim.obstacles:
        if len(obs) > 1:
            # Polygonal obstacle
            o = Polygon(obs)
            p = Point(rpos)
            p1, p2 = nearest_points(o, p)
            # Velocity is always 0 for obstacles
            # Heading is same as robot's
            state.extend([p1.x, p2.y, 0, 0, sim.obs_width])
        else:
            # Point obstacle
            state.extend([obs[0][0], obs[0][1], 0, 0, sim.obs_width])
    return state


def avg(vals):
    """Computes the average of the numerical values in `vals`.

    :param iterable vals: iterable of numerical values to average.
    :return: The numerical average of the values in `vals`
        :rtype: float
    """
    if len(vals) > 0:
        total = 0
        for v in vals:
            total += v
        return total / len(vals)
    return 0.0


def std_dev(vals):
    """Computes the (sample) standard deviation of the numerical values in
    `vals`.

    :param iterable vals: Numerical values to find the standard deviation of
    :return: The (sample) standard deviation of the values in `vals`
        :rtype: float
    """
    if len(vals) > 1:
        mean = avg(vals)
        total = 0
        for v in vals:
            total += (v - mean) ** 2
        return math.sqrt(total / (len(vals) - 1))
    return 0.0


def cov(x, y):
    """Computes the (sample) covariance between the values in x and y.

    :param iterable x: list of numerical values
    :param iterable y: list of numerical values (same len as x)
    :return: The (sample) covariance between the values in x and y
        :rtype: float
    """
    if len(x) == len(y) and len(x) > 1:
        x_bar = avg(x)
        y_bar = avg(y)
        acc = 0
        for i in range(len(x)):
            acc += (x[i] - x_bar) * (y[i] - y_bar)
        return acc / (len(x) - 1)
    return None


if __name__ == "__main__":
    main()
