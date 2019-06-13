""" simulator.py
Simulator the rl_model uses to train.
RISS 2019

"""

import pyrvo2.rvo2


class Simulator(object):

    def __init__(self, scene=None):
        pass

    def do_step(self, action_ind):
        pass

    def reward(self):
        pass

    def build_scene(self, scene):
        if scene == "barge_in":
            pass
        else:       # Build a random scene
            pass
