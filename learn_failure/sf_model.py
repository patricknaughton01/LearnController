"""sf_model.py
Run a simulation of a robot interacting with humans according to a social
forces model. The computation of forces is based on the forumla presented
in the paper "Robot Companion: A Social-Force based approach with Human
Awareness-Navigation in Crowded Environments"
(http://www.iri.upc.edu/files/scidoc/1458-Robot-Companion:-A -Social-Force
-based-approach-with-Human-Awareness-Navigation-in-Crowded-Environments.pdf)
"""

import utils
import simulator
import math
# Used to conveniently find the nearest point on a polygon to the robot
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points
from utils import dist


def main():
    args = utils.parse_args()
    scene = args.scene
    episodes = args.num_episodes
    max_timesteps = args.max_timesteps
    record = args.record
    epsilon = 10**-10   # Numerical stability
    # Assume robot is isotropic ==> don't need lambda term
    # Also assume robot has mass of 1
    # Robot-obstacle parameters
    aro, bro = 1.0, 1.0
    # Robot-pedestrian parameters
    arp, brp = 1.0, 1.0
    # Force weighting parameters
    gamma, delta = 1.5, 1.0
    for episode in range(episodes):
        print("Starting episode {}".format(episode))
        file = None
        if record:
            file = open("sf_trajectory_" + str(episode) + ".txt","w")
        sim = simulator.Simulator(scene=scene, file=file)
        for t in range(max_timesteps):
            # Calculate social force on robot
            fx, fy = 0, 0
            r_pos = sim.sim.getAgentPosition(sim.robot_num)
            r_rad = sim.sim.getAgentRadius(sim.robot_num)
            for agent in sim.agents:
                if agent != sim.robot_num:  # Robot doesn't exert force on self
                    a_pos = sim.sim.getAgentPosition(agent)
                    # Vector points from agent to robot
                    vx, vy = r_pos[0] - a_pos[0], r_pos[1] - a_pos[1]
                    d = r_rad + sim.sim.getAgentRadius(agent)
                    distance = dist((vx, vy), (0, 0)) + epsilon
                    mag = gamma * arp * math.exp((d - distance)/brp) / distance
                    fx += mag * vx
                    fy += mag * vy
            for obs in sim.obstacles:
                distance = 0
                if len(obs) > 1:
                    # Polygonal obstacle
                    o = Polygon(obs)
                    p = Point(r_pos)
                    p1, p2 = nearest_points(o, p)
                    o_point = (p1.x, p1.y)
                    if not o.contains(p):
                        distance = dist((p1.x, p1.y), (p2.x, p2.y))
                else:
                    # Point obstacle
                    distance = dist(r_pos, obs[0])
                    o_point = obs[0]
                vx, vy = r_pos[0] - o_point[0], r_pos[1] - o_point[1]
                distance += epsilon
                mag = delta * aro * math.exp((r_rad - distance)/bro) / distance
                fx += mag * vx
                fy += mag * vy
            ts = sim.sim.getTimeStep()
            r_vel = sim.sim.getAgentVelocity(sim.robot_num)
            r_vel = (r_vel[0] + ts * fx, r_vel[1] + ts * fy)
            mag = dist(r_vel, (0, 0))
            max_speed = sim.sim.getAgentMaxSpeed(sim.robot_num)
            # Don't let robot's velocity grow unbounded
            if mag > max_speed:
                r_vel = (max_speed * r_vel[0] / mag, max_speed * r_vel[1] /
                         mag)
            sim.goals[sim.robot_num] = (r_pos[0] + ts * r_vel[0], r_pos[1] +
                                        ts * r_vel[1])
            sim.advance_simulation()
        if record:
            file.close()


if __name__ == "__main__":
    main()
