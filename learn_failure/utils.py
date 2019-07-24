import numpy as np
import torch
import argparse
from functools import reduce
from operator import mul
from state import ObservableState

def parse_args():
    parser = argparse.ArgumentParser(description="Learn Controller")
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--print_every", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--visualize_val", action='store_true')
    parser.add_argument("--show_mc", action='store_true')
    parser.add_argument("--num", type=int, default=0)
    parser.add_argument("--kfold", type=int, default=5)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num_lstms", type=int, default=1)
    parser.add_argument("--dist_thres", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--tao", type=float, default=1.0)
    parser.add_argument("--length_scale", type=float, default=0)
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--exp", type=int, default=1)
    parser.add_argument("--bootstrap", action='store_true')
    parser.add_argument("--M", type=int, default=1)
    parser.add_argument("--train_data_name", type=str, default="simulate_crossing")
    parser.add_argument("--test_data_name", type=str, default="simulate_crossing")
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--model_type", type=str, default="crossing")
    parser.add_argument("--model_config", type=str, default="configs/model.config")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--test_percent", type=float, default=0.1)
    parser.add_argument("--video_name", type=str, default="")
    parser.add_argument("--scene", type=str, default="barge_in")
    parser.add_argument("--max_timesteps", type=int, default=10**8)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon_decay", type=float, default=0.9999907897)#0.99999769741)
    parser.add_argument("--target_update", type=int, default=10000)
    parser.add_argument("--converge_thresh", type=float, default=10**(-5))
    parser.add_argument("--record", type=bool, default=False)
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--success_path", type=str, default="")
    parser.add_argument("--success_max_ts", type=int, default=100)
    parser.add_argument("--name", type=str, default="")
    return parser.parse_args()

def get_weight_decay(tao, length_scale, N, dropout):
    return length_scale ** 2 * (1 - dropout) / (2. * N * tao)

def estimate_uncertainty(model, x, tao, T=100):
    model.train()
    preds = []
    for _ in range(T):
        pred = model(x)
        preds.append(pred.detach())
    preds = torch.stack(preds)[:, :, :, 0:2]
    mean_preds = torch.mean(preds, dim=0) # seq_len x batch_size x D
    base_x = x[0:1, :, 0:2]
    mean_pred_x = mean_preds[:, :, 0:2] + base_x
    pred_x = preds + base_x

    seq_len, batch_size, D = mean_pred_x.size()
    moment2 = torch.bmm(pred_x.view(-1, D).unsqueeze(-1), pred_x.view(-1, D).unsqueeze(1)).view(T, seq_len, batch_size, D, D)
    mean_moment2 = torch.mean(moment2, dim=0)

    mean_term = torch.bmm(mean_pred_x.view(-1, D).unsqueeze(-1), mean_pred_x.view(-1, D).unsqueeze(1)).view(seq_len, batch_size, D, D)

    # var_pred_x = mean_moment2 + 1.0 / tao * torch.eye(D).expand(seq_len, batch_size, D, D) - mean_term
    var_pred_x = mean_moment2 - mean_term

    return mean_pred_x, var_pred_x

def get_num_params(model):
    total = 0
    for param in model.parameters():
        if param.requires_grad:
            total += reduce(mul, param.size())
    return total

def partition(states, seq_lengths, num_humans, config):
    n = states.shape[0]
    indices = np.arange(n)
    np.random.shuffle(indices)
    test_percent = config.get('test_percent', 0.1)
    end = int(n * test_percent)

    val_states = states[0:end]
    val_seq_lengths = seq_lengths[0:end]
    val_num_humans = num_humans[0:end]

    train_states = states[end:]
    train_seq_lengths = seq_lengths[end:]
    train_num_humans = num_humans[end:]
    return (val_states, val_seq_lengths, val_num_humans), (train_states, train_seq_lengths, train_num_humans)

def covariance_to_std(var):
    temp = np.zeros((var.shape[0], var.shape[1], 2))
    temp[:, :, 0] = np.sqrt(var[:, :, 0, 0])
    temp[:, :, 1] = np.sqrt(var[:, :, 1, 1])
    return temp

def get_coefs(preds):
    mu_x, mu_y, sigma_x, sigma_y, corr = preds[:, :, 0], preds[:, :, 1], preds[:, :, 2], preds[:, :, 3], preds[:, :, 4]

    # print("mu_x.shape", mu_x.shape)
    # print("mu_y.shape", mu_y.shape)
    # print("sigma_x.shape", sigma_x.shape)
    # print("sigma_y.shape", sigma_y.shape)
    # print("mu_x.shape", mu_x.shape)

    # print("mu_x", mu_x)
    # print("mu_y", mu_y)
    # print("sigma_x", sigma_x)
    # print("sigma_y", sigma_y)

    # a = input('').split(" ")[0]
    # print(a)

    sigma_x = torch.exp(sigma_x)
    sigma_y = torch.exp(sigma_y)
    corr = torch.tanh(corr)
    return mu_x, mu_y, sigma_x, sigma_y, corr

def ensemble(preds, pred_xs):
    mean_pred_x = torch.mean(torch.stack(pred_xs), dim=0)

    seq_len, batch_size = mean_pred_x.size(0), mean_pred_x.size(1)
    var_pred_x = torch.zeros(seq_len, batch_size, 2, 2)

    data_uncertainty = 0
    for pred, pred_x in zip(preds, pred_xs):
        _, _, sigma_x, sigma_y, corr = get_coefs(pred)
        sigma = torch.cat((sigma_x, sigma_y), dim=-1) # concatenation - (-1) is same as 1
        # old shape = [17,16]
        # new shape = [136, 2, 1] (using the operation sigma.view(-1, 2, 1))
        sigma = torch.bmm(sigma.view(-1, 2, 1), sigma.view(-1, 1, 2)).view(seq_len, batch_size, 2, 2)
        sigma[:, :, 0, 1] *= corr
        sigma[:, :, 1, 0] *= corr
        data_uncertainty += sigma
        var_pred_x += sigma + torch.bmm(pred_x.view(-1, 2, 1), pred_x.view(-1, 1, 2)).view(seq_len, batch_size, 2, 2)

    data_uncertainty /= len(preds)
    var_pred_x /= len(preds)
    var_pred_x -= torch.bmm(mean_pred_x.view(-1, 2, 1), mean_pred_x.view(-1, 1, 2)).view(seq_len, batch_size, 2, 2)

    var_pred_x = var_pred_x.detach().cpu().numpy()
    data_uncertainty = data_uncertainty.detach().cpu().numpy()
    model_uncertainty = np.maximum(0, var_pred_x - data_uncertainty)
    mean_pred_x = mean_pred_x.detach().cpu().numpy()
    # print("mean_pred_x", mean_pred_x)
    return mean_pred_x, var_pred_x, data_uncertainty, model_uncertainty


def dataloader(data, config, is_train=True, shuffle=True):
    assert len(data) == 3, 'You should provide states, sequence lengths, and number of humans!'
    batch_size = config.get('batch_size', 16)
    bootstrap = config.get('bootstrap', False)
    states, seq_lengths, num_humans = data

    # print('batch size: %d' % batch_size)
    # print('data size: %d' % len(seq_lengths))
    # print('unique num of humans: ', np.unique(num_humans)) # 2.0, 3.0, 4.0
    # print('bootstrap', bootstrap)
    # print("states", states.shape)
    # print("seq_lengths", seq_lengths.shape)
    # print("num_humans", type(num_humans))
    # print("--------")
    for num_human in np.unique(num_humans):
        filt_indices = num_humans == num_human # selects all the matching number of humans

        filt_states = states[filt_indices] # only sucks out the required states (using above condition)

        filt_seq_lengths = seq_lengths[filt_indices] #same as above

        # print("filt_indices",filt_indices.shape)
        # print("filt_states",filt_states.shape)
        # print("filt_seq_lengths",filt_seq_lengths.shape)
        # print("seq_lengths[20]", seq_lengths[20])
        # the value of n is only the number of valid human states
        # so if there are 200/400 states with 2 humans, n = 200
        n = len(filt_seq_lengths)

        indices = np.arange(n) # generates an array with values from 0 to n
        if shuffle:
            np.random.shuffle(indices)
        idx = 0
        while idx < n:
            if bootstrap and is_train:
                selected_indices = np.random.choice(indices, size=batch_size, replace=True)
            else:
                selected_indices = indices[idx : idx + batch_size]
            # print("selected_indices shape", selected_indices.shape)

            # if i manually count it, it is = seq_lengths,
            # but if i print the size, it is = 8
            cur_states = filt_states[selected_indices]
            # print("cur_states", cur_states.shape)
            cur_size = cur_states.shape[0]
            # print("cur_size", cur_size)
            if cur_size > 0:
                # print("selected_indices", selected_indices)
                batch_seq_lengths = filt_seq_lengths[selected_indices] - 1
                # print("batch_seq_lengths", batch_seq_lengths)
                # print("filt_seq_lengths[selected_indices]", filt_seq_lengths[selected_indices])

                dim = cur_states[0].shape[-1]
                # print("dim", dim)
                # print("cur_states[0].shape", cur_states[0].shape)
                # print("cur_states[0].shape[-1]", cur_states[0].shape[-1])
                batch_states = np.zeros((max(batch_seq_lengths), cur_size, dim))
                # print("batch_states", batch_states)
                batch_future_states = np.zeros((max(batch_seq_lengths), cur_size, dim))
                batch_targets = np.zeros((batch_states.shape[0], cur_size, 2))

                for i, state in enumerate(cur_states):
                    seq_len = batch_seq_lengths[i]
                    batch_states[0:seq_len, i, :] = state[0:-1, :]

                    batch_future_states[0:seq_len, i, :] = state[1:, :]
                    #notice that here we are not giving
                    target = state[1:, 0:2] - state[0:-1, 0:2]
                    # print("state[1:, 0:2]", state[1:, 0:2].shape)
                    # print("state[0:-1, 0:2]", state[0:-1, 0:2].shape)

                    batch_targets[0:seq_len, i, :] = target

                yield batch_states, batch_seq_lengths, batch_targets, batch_future_states
                idx += cur_size
            else:
                break

def rotate(state, kinematics='unicycle'):
    """
    Transform the coordinate to agent-centric.
    Input state tensor is of size (batch_size, state_length)
    """
    # 'px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta', 'px1', 'py1', 'vx1', 'vy1', 'radius1'
    #  0     1      2     3      4        5     6      7         8       9     10      11     12       13
    batch = state.shape[0]
    dx = (state[:, 5] - state[:, 0]).reshape((batch, -1))
    dy = (state[:, 6] - state[:, 1]).reshape((batch, -1))
    rot = torch.atan2(state[:, 6] - state[:, 1], state[:, 5] - state[:, 0])

    dg = torch.norm(torch.cat([dx, dy], dim=1), 2, dim=1, keepdim=True)
    v_pref = state[:, 7].reshape((batch, -1))
    vx = (state[:, 2] * torch.cos(rot) + state[:, 3] * torch.sin(rot)).reshape((batch, -1))
    vy = (state[:, 3] * torch.cos(rot) - state[:, 2] * torch.sin(rot)).reshape((batch, -1))

    radius = state[:, 4].reshape((batch, -1))
    if kinematics == 'unicycle':
        theta = (state[:, 8] - rot).reshape((batch, -1))
    else:
        # set theta to be zero since it's not used
        theta = torch.zeros_like(v_pref)
    vx1 = (state[:, 11] * torch.cos(rot) + state[:, 12] * torch.sin(rot)).reshape((batch, -1))
    vy1 = (state[:, 12] * torch.cos(rot) - state[:, 11] * torch.sin(rot)).reshape((batch, -1))
    px1 = (state[:, 9] - state[:, 0]) * torch.cos(rot) + (state[:, 10] - state[:, 1]) * torch.sin(rot)
    px1 = px1.reshape((batch, -1))
    py1 = (state[:, 10] - state[:, 1]) * torch.cos(rot) - (state[:, 9] - state[:, 0]) * torch.sin(rot)
    py1 = py1.reshape((batch, -1))
    radius1 = state[:, 13].reshape((batch, -1))
    radius_sum = radius + radius1
    da = torch.norm(torch.cat([(state[:, 0] - state[:, 9]).reshape((batch, -1)), (state[:, 1] - state[:, 10]).
                                reshape((batch, -1))], dim=1), 2, dim=1, keepdim=True)
    new_state = torch.cat([dg, v_pref, theta, radius, vx, vy, px1, py1, vx1, vy1, radius1, da, radius_sum], dim=1)
    return new_state

def transform_and_rotate(raw_states):
    # states shape: batch_size x dim
    # 'px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta', 'px1', 'py1', 'vx1', 'vy1', 'radius1', ..., 'radiusN'
    #  0     1      2     3      4        5     6      7         8       9     10      11     12       13 , ...,
    num_human = int((raw_states.shape[1] - 9) / 5)
    self_state = raw_states[:, 0:9]
    # print("self_state.shape", self_state.shape)
    human_states = [raw_states[:, 9 + i * 5 : 9 + (i + 1) * 5] for i in range(num_human)]
    # print("human_states.shape", len(human_states))
    # print("human_states[0]",human_states[0].shape)

    cur_states = torch.stack([torch.Tensor(np.concatenate((self_state, human_state), axis=1)) for human_state in human_states])
    # print("cur_states.shape", cur_states.shape)
    cur_states = cur_states.transpose(0, 1).contiguous()
    # print("cur_states.shape new", cur_states.shape)
    # now states size is: batch_size x num_human x dim

    batch_size, dim = cur_states.size(0), cur_states.size(2)

    rotated_states = rotate(cur_states.view(-1, dim)).view(batch_size, num_human, -1)
    # print("cur_states.view(-1, dim).shape", cur_states.view(-1, dim).shape)
    return rotated_states # [8, 6, 13]

def build_humans(states):
    #state shape: dim
    num_human = int((states.shape[0] - 4) / 5)
    human_states = []
    for i in range(num_human):
        if i == 0:
            px = states[0]
            py = states[1]
            vx = states[2]
            vy = states[3]
            radius = states[4]
        else:
            px = states[9 + (i - 1) * 5]
            py = states[10 + (i - 1) * 5]
            vx = states[11 + (i - 1) * 5]
            vy = states[12 + (i - 1) * 5]
            radius = states[13 + (i - 1) * 5]
        human_states.append(ObservableState(px, py, vx, vy, radius))
    return human_states

def build_occupancy_maps(human_states, config={}):
    """Builds an occupancy map for each human in human_states.

    If `om_channel_size` is 1, each occupancy map is simply a grid centered on
    a given human and an indication in each of the `cell_num**2` cells
    of whether or not it is occupied.

    If `om_channel_size` is 2, the final occupancy map is `2 * cell_num**2`
    elements and each pair of elements is the average x and y velocities of
    agents in that cell respectively.

    If `om_channel_size` is 3, the final occupancy map is `3 * cell_num**2`
    elements and each triplet is 1 if there's at least one agent in that cell
    and the other two elements are the same as for if `om_channel_size` is 2
    (i.e. the average velocities).

    :param human_states: State of all humans in the scene (Note, as we
        are using it, this includes the robot itself and the points on any
        obstacles which are closest to the robot. The format is
        [x, y, vx, vy, rad]).
    :param config: Configuration for the function to specify:
        om_channel_size
        cell_num
        cell_size
    :return: tensor of shape (# human - 1, self.cell_num ** 2)
    """
    om_channel_size = config.get('om_channel_size', 3)
    cell_num = config.get('cell_num', 4)
    cell_size = config.get('cell_size', 1.0)
    occupancy_maps = []
    for human in human_states:
        other_humans = np.concatenate([np.array([(other_human.px, other_human.py, other_human.vx, other_human.vy)])
                                        for other_human in human_states if other_human != human], axis=0)
        other_px = other_humans[:, 0] - human.px
        other_py = other_humans[:, 1] - human.py
        # new x-axis is in the direction of human's velocity
        human_velocity_angle = np.arctan2(human.vy, human.vx)
        other_human_orientation = np.arctan2(other_py, other_px)
        rotation = other_human_orientation - human_velocity_angle
        distance = np.linalg.norm([other_px, other_py], axis=0)
        other_px = np.cos(rotation) * distance
        other_py = np.sin(rotation) * distance

        # compute indices of humans in the grid
        other_x_index = np.floor(other_px / cell_size + cell_num / 2)
        other_y_index = np.floor(other_py / cell_size + cell_num / 2)
        other_x_index[other_x_index < 0] = float('-inf')
        other_x_index[other_x_index >= cell_num] = float('-inf')
        other_y_index[other_y_index < 0] = float('-inf')
        other_y_index[other_y_index >= cell_num] = float('-inf')
        grid_indices = cell_num * other_y_index + other_x_index
        occupancy_map = np.isin(range(cell_num ** 2), grid_indices)
        if om_channel_size == 1:
            occupancy_maps.append([occupancy_map.astype(int)])
        else:
            # calculate relative velocity for other agents
            other_human_velocity_angles = np.arctan2(other_humans[:, 3], other_humans[:, 2])
            rotation = other_human_velocity_angles - human_velocity_angle
            speed = np.linalg.norm(other_humans[:, 2:4], axis=1)
            other_vx = np.cos(rotation) * speed
            other_vy = np.sin(rotation) * speed
            dm = [list() for _ in range(cell_num ** 2 * om_channel_size)]
            for i, index in np.ndenumerate(grid_indices):
                if index in range(cell_num ** 2):
                    if om_channel_size == 2:
                        dm[2 * int(index)].append(other_vx[i])
                        dm[2 * int(index) + 1].append(other_vy[i])
                    elif om_channel_size == 3:
                        dm[2 * int(index)].append(1)
                        dm[2 * int(index) + 1].append(other_vx[i])
                        dm[2 * int(index) + 2].append(other_vy[i])
                    else:
                        raise NotImplementedError
            for i, cell in enumerate(dm):
                dm[i] = sum(dm[i]) / len(dm[i]) if len(dm[i]) != 0 else 0
            occupancy_maps.append([dm])

    return torch.from_numpy(np.concatenate(occupancy_maps, axis=0)).float()

def neg_2d_gaussian_likelihood(outputs, targets, seq_lengths):
    # Extract mean, std devs and correlation
    mux, muy, sx, sy, corr = get_coefs(outputs)

    # Compute factors
    normx = targets[:, :, 0] - mux
    normy = targets[:, :, 1] - muy
    sxsy = sx * sy

    z = (normx/sx)**2 + (normy/sy)**2 - 2*((corr*normx*normy)/sxsy)
    negRho = 1 - corr**2

    # Numerator
    result = torch.exp(-z/(2*negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

    # Numerical stability
    epsilon = 1e-10

    # Final PDF calculation
    result = result / (denom + epsilon)

    result = -torch.log(result + epsilon)
    mask = []
    seq_len, batch_size = result.size()
    error_vector = targets - outputs[:, :, 0:2]
    error = torch.norm(error_vector, p=2, dim=-1)
    for i in range(batch_size):
        mask.append(torch.arange(0, seq_len) < seq_lengths[i])
        # print("val1", torch.arange(0, seq_len) )
        # print("val2", seq_lengths[i])
        # print("val3", torch.arange(0, seq_len) < seq_lengths[i])
    mask = torch.stack(mask).transpose(0, 1).float()
    # print("mask", mask)
    total = torch.sum(mask)
    # print("total", total)

    return torch.sum(result * mask) / total, torch.sum(error * mask) / total

def rotate(state, kinematics='unicycle'):
    """
    Transform the coordinate to agent-centric.
    Input state tensor is of size (batch_size, state_length)
    """
    # 'px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta', 'px1', 'py1', 'vx1', 'vy1', 'radius1'
    #  0     1      2     3      4        5     6      7         8       9     10      11     12       13
    batch = state.shape[0]
    dx = (state[:, 5] - state[:, 0]).reshape((batch, -1))
    dy = (state[:, 6] - state[:, 1]).reshape((batch, -1))
    rot = torch.atan2(state[:, 6] - state[:, 1], state[:, 5] - state[:, 0])

    dg = torch.norm(torch.cat([dx, dy], dim=1), 2, dim=1, keepdim=True)
    v_pref = state[:, 7].reshape((batch, -1))
    vx = (state[:, 2] * torch.cos(rot) + state[:, 3] * torch.sin(rot)).reshape((batch, -1))
    vy = (state[:, 3] * torch.cos(rot) - state[:, 2] * torch.sin(rot)).reshape((batch, -1))

    radius = state[:, 4].reshape((batch, -1))
    if kinematics == 'unicycle':
        theta = (state[:, 8] - rot).reshape((batch, -1))
    else:
        # set theta to be zero since it's not used
        theta = torch.zeros_like(v_pref)
    vx1 = (state[:, 11] * torch.cos(rot) + state[:, 12] * torch.sin(rot)).reshape((batch, -1))
    vy1 = (state[:, 12] * torch.cos(rot) - state[:, 11] * torch.sin(rot)).reshape((batch, -1))
    px1 = (state[:, 9] - state[:, 0]) * torch.cos(rot) + (state[:, 10] - state[:, 1]) * torch.sin(rot)
    px1 = px1.reshape((batch, -1))
    py1 = (state[:, 10] - state[:, 1]) * torch.cos(rot) - (state[:, 9] - state[:, 0]) * torch.sin(rot)
    py1 = py1.reshape((batch, -1))
    radius1 = state[:, 13].reshape((batch, -1))
    radius_sum = radius + radius1
    da = torch.norm(torch.cat([(state[:, 0] - state[:, 9]).reshape((batch, -1)), (state[:, 1] - state[:, 10]).
                                reshape((batch, -1))], dim=1), 2, dim=1, keepdim=True)
    new_state = torch.cat([dg, v_pref, theta, radius, vx, vy, px1, py1, vx1, vy1, radius1, da, radius_sum], dim=1)
    return new_state

def dist(p1, p2):
    """Calculate the distance between p1 and p2

    :param tuple p1: (x, y) of p1.
    :param tuple p2: (x, y) of p2.
    :return: The Euclidean distance between the two points.
        :rtype: float

    """
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
