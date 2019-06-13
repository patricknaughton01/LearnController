import logging
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import simulator
from torch.autograd import Variable
from torch.utils.data import DataLoader
from time import time
from utils import neg_2d_gaussian_likelihood, transform_and_rotate, build_occupancy_maps, build_humans

class Trainer(object):
    def __init__(self, model, config):
        """
        Train the trainable model of a policy
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

    def run(self):
        # print('in run')
        num_episodes = self.config.get('num_episodes', 100)
        print_every = self.config.get('print_every', 10)
        log_path = self.config.get('log_path', 'log')
        for episode in range(1, num_episodes + 1):
            run_episode()
            print("Ran episode {}".format(episode))

    def run_episode(self):
        sim = simulator.Simulator(scene="barge_in")
        h_t = None
        for i in range(self.max_timesteps):
            action = self.policy_model.select_action()

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
        transitions = memory.sample(self.batch_size)
        # Transpose the batch i.e. turn a batch of transitions into a
        # transition of batch arrays
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        h_t_batch = torch.cat(batch.h_t)
        action_batch = torch.cat(batch.action)
        next_state_batch = torch.cat(batch.next_state)
        reward_batch = torch.cat(batch.reward)

        # Compute the Q values we expected
        state_action_values = policy_model(state_batch, h_t_batch).gather(1,
            action_batch)
        # Compute expected Q values based on Bellman equation
        next_state_values = target_model(next_state_batch,
            h_t_batch).max(1)[0].detach()
        expected_state_action_vals = ((next_state_values * self.gamma)
            + reward_batch)
        # Compute loss
        loss = F.smooth_l1_loss(state_action_values,
            expected_state_action_vals.unsqueeze(1))
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# Store transitions from state to next_state via action and what reward
# we received as well as the hidden state at that point
Transition = namedtuple('Transition',
    ('state', 'h_t', 'action', 'next_state', 'reward'))


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
