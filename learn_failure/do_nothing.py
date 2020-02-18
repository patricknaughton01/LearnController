"""do_nothing.py
Run simulations of the robot in various scenarios to see how it performs
when it simply does nothing.
"""

import simulator
import utils
import time
import random


def main():
    args = utils.parse_args()
    max_timesteps = args.max_timesteps
    num_episodes = args.num_episodes
    scene = args.scene
    if args.seed is None:
        args.seed = time.time()
    random.seed(args.seed)
    for episode in range(num_episodes):
        file = None
        if args.record:
            file = open("do_nothing_" + str(episode) + ".txt", "w")
        sim = simulator.Simulator(scene=scene, file=file)
        sim.sim.setAgentMaxSpeed(0, 0)
        for t in range(max_timesteps):
            # Just to be safe, always set robot goal to its current position
            sim.goals[sim.robot_num] = sim.sim.getAgentPosition(sim.robot_num)
            sim.advance_simulation()
        if args.record:
            file.close()

if __name__ == "__main__":
    main()
