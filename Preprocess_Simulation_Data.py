#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
import math
import re
from matplotlib import patches
import numpy as np
from glob import glob
import time
import warnings
warnings.filterwarnings("ignore")
import os

obstacles = []
# In[2]:

def main():
    global obstacles
    parser = argparse.ArgumentParser(description="Pre-process simulation data")
    parser.add_argument('--animate', action="store_true",
                        help="animate the playback")
    parser.add_argument('--save', action="store_true",
                        help="save the animation")
    parser.add_argument('--animation_name', type=str, default="animation.mp4",
                        help="filename to save the animation as")
    parser.add_argument("--data_type", type=str, default="sf_trajectory",
                        help="filename prefix to display")
    parser.add_argument("--directory", type=str,
                       default="learn_failure/tests/social_forces"
                               "/dynamic_barge_in_test_1/",
                       help="directory to find the trajectory files.")
    parser.add_argument("--stride", type=int, default=1, help="how many "
                                                              "frames to "
                                                              "advance each "
                                                              "time")
    parser.add_argument("--i", type=int, default=0, help="index of "
                                                         "trajectory to "
                                                         "display")
    args = parser.parse_args()
    data_type = args.data_type
    directory = args.directory
    parent_path = directory + "/" + data_type + '_*.txt'
    print(parent_path)
    if args.stride < 1:
        args.stride = 1

    # In[3]:

    states = []
    seq_lengths = []
    files = glob(parent_path)
    files.sort(key=lambda x: int(x.split('/')[-1].split('.')[-2].split('_')[-1]))
    obstacle_strs = []
    for path in files:
        # print(path)
        if 'obstacles' in path:
            continue
        # print('preprocessing %s' % path)
        with open(path, 'r') as file:
            cur_states = []
            obstacle_strs.append(file.readline().strip())
            lines = file.readlines()
            for i, line in enumerate(lines):
                # print('line')
                # print(line)

                data = line.split()
                # print('data')
                # print(data)

                if i == 0:
                    # print('first data')
                    # print(data)
                    continue
                    # the first line contains only the headers
                state = []

                for j, data in enumerate(line.split()):
                    # print ('data is %s' % data )
                    if j == 0:
                        continue
                        # we dont care about the timestamp

                    elif '(' in data:
                        x, y = data.lstrip('(').rstrip(')').split(',')
                        state.extend([float(x), float(y)])
                    else:
                        state.append(float(data))
                    # the way extend and append is being used here, everything gets added serially
                    # [ a, f, v, vf, vfv ...]
                cur_states.append(state)
                # print(len(state)) = 14 constantly
                # print('state')
                # print(state)
                # print('len(cur_states)')
                # print(len(cur_states))
                # print(cur_states)
            seq_lengths.append(len(cur_states))
            # this is append the total num of lines that were present in the data file
            # print('seq_lengths')
            # print(seq_lengths)
            states.append(cur_states)
            # print(len(states))


    # In[4]:

    n = len(seq_lengths) #will store the number of files
    # print(n)
    seq_lengths = np.array(seq_lengths)
    num_humans = np.zeros((n, ))
    states_array = np.empty((n, ), dtype=object)
    for i, state in enumerate(states):
        states_array[i] = np.array(state)
        # gets the same value that was previously held by cur_states
        # i think this can be done by assigning cur_states also instead of putting
        # this in a diff loop
        # print(states_array[i])
        # print('\n')
        # print(len(state[0]))
        num_humans[i] = (len(state[0]) - 4) / 5
        # print(num_humans[i])


    # In[5]:


    save_path = 'learn_failure/simulate_' + data_type.lower()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print('save path is: %s' % save_path)


    # In[7]:


    # obstacle_files = glob(directory  + data_type + "_obstacles_*.txt")
    # obstacle_files.sort(key=lambda x: int(x.split('/')[-1].split('.')[-2].split('_')[-1]))

    filt = seq_lengths < 400
    # print(filt)
    # print(seq_lengths)
    seq_lengths = seq_lengths[filt]
    # print(seq_lengths[filt])
    # print(num_humans)
    num_humans = num_humans[filt]
    # print(num_humans[filt])
    states_array = states_array[filt]
    # print(states_array)
    # n = states_array.shape[0]
    # filt = np.empty((n, ), dtype=bool)
    # filt.fill(True)


    # hist = [100, 100, 100, 100, 100, 100]
    # for i in range(n):
    #     if np.mean(np.abs(states_array[i][2:, 8] - states_array[i][1, 8])) >= 0.3:
    #         filt[i] = False
    #         hist[i//100] -= 1

    # seq_lengths = seq_lengths[filt]
    # num_humans = num_humans[filt]
    # states_array = states_array[filt]

    # for i in range(np.where(filt==False)[0])
    #     os.remove(directory  + data_type + "_obstacles_%d.txt" % i)


    # In[8]:


    np.save(save_path + '/seq_lengths.npy', seq_lengths)
    # print(seq_lengths)
    np.save(save_path + '/states.npy', states_array)
    # print(states_array)
    np.save(save_path + '/num_humans.npy', num_humans)
    # print(num_humans)


    # In[9]:


    seq_lengths = np.load(save_path + '/seq_lengths.npy',allow_pickle=True)
    states_array = np.load(save_path + '/states.npy',allow_pickle=True)
    num_humans = np.load(save_path + '/num_humans.npy',allow_pickle=True)


    # In[20]:


    # visualization of the data
    idx = args.i
    obs_str = obstacle_strs[idx]

    episode = states_array[idx]
    n = int((episode.shape[1] - 4) / 6)
    print(n)
    colors = [np.array([1, 0, 0])]
    for i in range(n-1):
        colors.append(np.array([0, 0, i/(n-1)]))
    # colors = [np.random.rand(3)]
    print(colors)

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    ax.axis('equal')
    T = episode.shape[0]

    matches = re.findall(r"\[[^\[\]]+?\]", obs_str)
    # Build up the obstacles from sets of points
    for m in matches:
        points = re.findall(r"\(([0-9\-.+e]+, [0-9\-.+e]+)\)", m)
        obstacle = []
        for p in points:
            tmp = p.split(", ")
            obstacle.append((float(tmp[0]), float(tmp[1])))
        obstacles.append(obstacle)
    obstacle_color = (0.0, 0.0, 0.0)
    for i in range(len(obstacles)):
        ax.add_patch(patches.Polygon(
            obstacles[i], closed=True, color=obstacle_color,
            fill=False, linewidth=2.0
        ))
    # plt.hold(True)

    # ## check if obstacle exists
    # obstacle_files = glob(directory  + data_type + "_obstacles_*.txt")
    # obstacle_path = directory  + data_type + "_obstacles_%d.txt" % idx
    # if obstacle_path in obstacle_files:
    #     obstacles = np.loadtxt(obstacle_path)
    #     num_vertices = int(obstacles.shape[1] / 2)
    #     num_obstacles = obstacles.shape[0]
    #     for j in range(num_obstacles):
    #         for k in range(num_vertices):
    #             plt.plot(obstacles[j, [k * 2, (k * 2 + 2) % obstacles.shape[1]]],
    #                      obstacles[j, [1 + 2 * k, (3 + 2 * k) % obstacles.shape[1]]], 'k-')
    if args.animate:
        ani = animation.FuncAnimation(fig, animate, interval=100,
                                      fargs=(fig, episode, colors,
                                             -5, 5, 5, -5), repeat=True,
                                      frames=episode.shape[0])
        if args.save:
            ani.save(args.animation_name)
    else:
        for t in range(0, T, args.stride):
            for i in range(n):
                if i == 0:
                    x_idx, y_idx = 0, 1
                    radius = episode[0, 4]
                    heading = episode[t, 5]
                else:
                    x_idx, y_idx = 10 + (i-1) * 6, 10 + (i-1) * 6 + 1
                    radius = episode[0, 14 + (i-1) * 6]
                    heading = episode[t, 15 + (i - 1) * 6]
                # print('radius is %.4f' % radius)
                plt.plot(episode[:, x_idx], episode[:, y_idx], '-.', color=colors[i])

                e = patches.Ellipse((episode[t, x_idx], episode[t, y_idx]), radius * 2, radius * 2, linewidth=2, fill=False, zorder=2, color=colors[i])
                ax.add_patch(e)
                hx = radius * math.cos(heading) + episode[t, x_idx]
                hy = radius * math.sin(heading) + episode[t, y_idx]
                heading_rad = 0.05
                h = patches.Ellipse((hx, hy), heading_rad * 2, heading_rad * 2,
                                    linewidth=2, color=colors[i], fill=False)
                ax.add_patch(h)


    legends = ['agent %d' % i for i in range(n)]
    # max_lim = np.max(episode)
    # min_lim = np.min(episode)
    # plt.xlim((min_lim, max_lim))
    # plt.ylim((min_lim, max_lim))
    plt.legend(legends)
    plt.show()


def animate(frame, fig, episode, colors, left, right, top, bottom):
    global obstacles
    ax = fig.get_axes()[0]
    ax.axis('equal')
    ax.clear()
    obstacle_color = (0.0, 0.0, 0.0)
    for i in range(len(obstacles)):
        ax.add_patch(patches.Polygon(
            obstacles[i], closed=True, color=obstacle_color,
            fill=False, linewidth=2.0
        ))
    ax.set_xlim((left, right))
    ax.set_ylim((bottom, top))
    n = int((episode.shape[1] - 4) / 6)
    for i in range(n):
        if i == 0:
            x_idx, y_idx = 0, 1
            radius = max(episode[0, 4], 0.01)
            heading = episode[frame, 5]
        else:
            x_idx, y_idx = 10 + (i - 1) * 6, 10 + (i - 1) * 6 + 1
            radius = max(episode[0, 14 + (i - 1) * 6], 0.01)
            heading = episode[frame, 15 + (i - 1) * 6]
        # print('radius is %.4f' % radius)
        #plt.plot(episode[:, x_idx], episode[:, y_idx], '-.', color=colors[i])

        e = patches.Ellipse((episode[frame, x_idx], episode[frame, y_idx]), radius * 2,
                            radius * 2, linewidth=2, fill=False, zorder=2,
                            color=colors[i])
        hx = radius * math.cos(heading) + episode[frame, x_idx]
        hy = radius * math.sin(heading) + episode[frame, y_idx]
        heading_rad = 0.05
        h = patches.Ellipse((hx, hy), heading_rad * 2, heading_rad * 2,
                            linewidth=2, color=colors[i], fill=False)
        ax.add_patch(e)
        ax.add_patch(h)

if __name__ == "__main__":
    main()
