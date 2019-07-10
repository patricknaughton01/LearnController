"""policy.py
A representation of the policy for a reinforcement learning agent.
"""
import torch.nn as nn


class Policy(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)


    def forward(self, state, h_t):
        pass
