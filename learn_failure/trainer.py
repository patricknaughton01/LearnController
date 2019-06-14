import logging
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import simulator
import random
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
from time import time
from collections import namedtuple
from utils import neg_2d_gaussian_likelihood, transform_and_rotate, build_occupancy_maps, build_humans

class Trainer(object):
    def __init__(self, model, config):
        """Train the trainable model of a policy

        """
        # Policy model is used to select actions, target_model is used to
        # evaluate Q when we make our training updates. This has been shown
        # to increase stability of training.
        self.policy_model = model
        self.target_model = copy.deepcopy(model)
        self.criterion = neg_2d_gaussian_likelihood
        self.optimizer = optim.Adam(model.parameters(),
                                    lr=config.get('learning_rate', 0.001),
                                    weight_decay=config.get('weight_decay', 0))
        self.config = config
        # TODO: push the following things into the config
        self.batch_size = 32
        self.memory = ReplayMemory(1000000)
        self.gamma = 0.9
        self.max_timesteps = 128
        self.epsilon = 0.9
        self.epsilon_decay = 0.9998

    def run(self):
        """Run the trainer to train the policy_model such that it learns
        an optimal policy with respect to the reward given by the simulator.

        :return:
            :rtype: None

        """
        # print('in run')
        num_episodes = self.config.get('num_episodes', 100)
        print_every = self.config.get('print_every', 10)
        log_path = self.config.get('log_path', 'log')
        for episode in range(1, num_episodes + 1):
            reward = self.run_episode(record=True, key=episode)
            print("Ran episode {}\n\tGot reward {}".format(
                episode, reward[0][0]
            ))

    def run_episode(self, record=False, key=0):
        """Run one episode of training by creating a simulation using
        RVO2 (specifically the `Simulator` class in the `simulator` module.
        Episodes are limited to `self.max_timesteps` timesteps.

        :param boolean record: Whether or not to record this episode in a file
        :param int key: The number to append to this file to make it unique

        :return: 1x1 tensor containing total reward earned in this episode
            :rtype: tensor

        """
        out_file = None
        if record:
            out_file = open(str("barge") + "_" + str(key) + ".txt", "w")
        sim = simulator.Simulator(scene="barge_in", file=out_file)
        h_t = None
        curr_state = sim.state()
        total_reward = torch.zeros((1, 1), dtype=torch.float)
        for i in range(self.max_timesteps):
            action, h_t = self.policy_model.select_action(
                sim.state(), h_t, epsilon=self.epsilon
            )
            self.epsilon *= self.epsilon_decay
            sim.do_step(action)
            reward, _ = sim.reward()
            next_state = sim.state()
            if h_t is not None:
                self.memory.push(
                    curr_state, h_t[0], h_t[1], action, next_state, reward
                )
            else:
                self.memory.push(
                    curr_state, torch.zeros((1, 256)), torch.zeros((1, 256)),
                    action, next_state, reward
                )
            curr_state = next_state
            total_reward += reward
            self.optimize_model()
        if record:
            out_file.close()
        return total_reward

    def optimize_model(self):
        """Does one step of the optimization of the policy_model by sampling
        a batch of transitions and pushing the estimated Q values closer to
        their true values using the Bellman expectation equation and using the
        target_model to estimate the values of the next states.

        :return:
            :rtype: None

        """
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)

        total_loss = Variable(torch.zeros(1), requires_grad=True).clone()
        for transition in transitions:
            state_action_value = self.policy_model(
                transition.state, (transition.h_t, transition.c_t)
            )[0].gather(1, transition.action)  #?
            next_state_value = self.target_model(
                transition.next_state, (transition.h_t, transition.c_t)
            )[0].max(1)[0].detach()
            expected_state_action_val = ((next_state_value * self.gamma)
                + transition.reward)
            total_loss += F.smooth_l1_loss(
                state_action_value, expected_state_action_val
            ) / len(transitions)
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()


# Store transitions from state to next_state via action and what reward
# we received as well as the hidden state at that point
Transition = namedtuple('Transition',
    ('state', 'h_t', 'c_t', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    """Class to store a memory of our most recent transitions. This is sampled
    from to train our network to recognize q values. This sampling helps
    reduce correlations between transitions (which would be incredibly strong
    if sampled sequentially) to make training more robust and less likely to
    diverge.
    """
    def __init__(self, capacity):
        """Initialize the replay memory

        Parameters
        ----------
        capacity : int
            Maximum number of transitions to hold

        Returns
        -------
        None

        """
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Push a transition into the memory. If we have reached capacity,
        replace the oldest transition

        Parameters
        ----------
        *args : Transition(state, action, next_state, reward)
            The four components necessary to make a transition

        Returns
        -------
        None

        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Sample `batch_size` elements from the memory uniformly randomly

        Parameters
        ----------
        batch_size : int
            Number of elements to sample.

        Returns
        -------
        tuple
            List of transitions (sampled uniformly randomly)

        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
