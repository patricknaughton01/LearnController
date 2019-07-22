#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
from matplotlib import patches
import numpy as np
from glob import glob
import time
import warnings
warnings.filterwarnings("ignore")
import os


# In[2]:

def main():
    data_type = 'dynamic_barge_in'
    directory = 'learn_failure/'
    parent_path = directory  + data_type + '_*.txt'
    print(parent_path)

    parser = argparse.ArgumentParser(description="Pre-process simulation data")
    parser.add_argument('--animate', type=bool, default=False,
                        help="animate the playback")
    parser.add_argument('--save', type=bool, default=False,
                        help="save the animation")
    parser.add_argument('--animation_name', type=str, default="animation.mp4",
                        help="filename to save the animation as")
    args = parser.parse_args()

    # In[3]:

    states = []
    seq_lengths = []
    files = glob(parent_path)
    files.sort(key=lambda x: int(x.split('/')[-1].split('.')[-2].split('_')[-1]))
    for path in files:
        # print(path)
        if 'obstacles' in path:
            continue
        # print('preprocessing %s' % path)
        with open(path, 'r') as file:
            cur_states = []
            for i, line in enumerate(file):
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


    save_path = '/home/patricknaughton01/Downloads/LearnControllers/learn_failure/simulate_' + data_type.lower()
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
    idx = 0

    episode = states_array[idx]
    n = int((episode.shape[1] - 4) / 5)
    print(n)
    colors = [np.random.rand(3) for _ in range(n)]
    # colors = [np.random.rand(3)]
    print(colors)

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    ax.axis('equal')
    T = episode.shape[0]
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
        for t in range(T):
            for i in range(n):
                if i == 0:
                    x_idx, y_idx = 0, 1
                    radius = episode[0, 4]
                else:
                    x_idx, y_idx = 9 + (i-1) * 5, 9 + (i-1) * 5 + 1
                    radius = episode[0, 13 + (i-1) * 5]
                # print('radius is %.4f' % radius)
                plt.plot(episode[:, x_idx], episode[:, y_idx], '-.', color=colors[i])

                e = patches.Ellipse((episode[t, x_idx], episode[t, y_idx]), radius * 2, radius * 2, linewidth=2, fill=False, zorder=2, color=colors[i])
                ax.add_patch(e)


    legends = ['agent %d' % i for i in range(n)]
    # max_lim = np.max(episode)
    # min_lim = np.min(episode)
    # plt.xlim((min_lim, max_lim))
    # plt.ylim((min_lim, max_lim))
    plt.legend(legends)
    plt.show()


def animate(frame, fig, episode, colors, left, right, top, bottom):
    ax = fig.get_axes()[0]
    ax.axis('equal')
    ax.clear()
    ax.set_xlim((left, right))
    ax.set_ylim((bottom, top))
    n = int((episode.shape[1] - 4) / 5)
    for i in range(n):
        if i == 0:
            x_idx, y_idx = 0, 1
            radius = episode[0, 4]
        else:
            x_idx, y_idx = 9 + (i - 1) * 5, 9 + (i - 1) * 5 + 1
            radius = episode[0, 13 + (i - 1) * 5]
        # print('radius is %.4f' % radius)
        #plt.plot(episode[:, x_idx], episode[:, y_idx], '-.', color=colors[i])

        e = patches.Ellipse((episode[frame, x_idx], episode[frame, y_idx]), radius * 2,
                            radius * 2, linewidth=2, fill=False, zorder=2,
                            color=colors[i])
        ax.add_patch(e)

if __name__ == "__main__":
    main()
