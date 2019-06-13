""" simulator.py
Simulator the rl_model uses to train.
RISS 2019

"""

import pyrvo2.rvo2 as rvo2
import math
import utils
# Used to convieniently find the nearest point on a polygon to the robot
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points


class Simulator(object):

    def __init__(self, scene=None):
        self.sim = rvo2.PyRVOSimulator()
        self.obstacles = []
        self.robot_num = None
        self.agents = []
        self.scene = scene
        self.obs_width = 0.3
        self.build_scene(scene)

    def do_step(self, action_ind):
        """Run a step of the simulation.

        :param int action_ind: The index of the action to take.
        :return:
            :rtype: None

        """
        robot_max_vel = self.sim.getAgentMaxSpeed(self.robot_num)
        # Decode the action selection:
        #   0 => do nothing
        #   1-16 => set velocity to `robot_max_vel/2` at angle
        #       `(action_ind-1) * 2pi/16`
        #   17-33 => velocity to `robot_max_vel` at angle
        #       `(action_ind-17) * 2pi/16`
        #   else => do nothing
        vel = (0, 0)
        if action_ind >= 1 and action_ind <= 16:
            vel = (
                (robot_max_vel/2) * math.cos((action_ind - 1)*(math.pi / 8)),
                (robot_max_vel/2) * math.sin((action_ind - 1)*(math.pi / 8))
            )
        elif action_ind >= 17 and action_ind <= 33:
            vel = (
                robot_max_vel * math.cos((action_ind - 17)*(math.pi / 8)),
                robot_max_vel * math.sin((action_ind - 17)*(math.pi / 8))
            )
        self.sim.setAgentVelocity(self.robot_num, vel)
        self.sim.doStep()

    def state(self):
        """Compute the state that will be fed to the DQN. This is composed
        of the robot's rotated and transformed state (see utils.py / ask
        Yash for what that does) and the occupancy map (same advice for how
        this is built).

        :return: The tensor that can be given to the DQN's value estimation
            network.
            :rtype: tensor

        """
        rpos = self.sim.getAgentPosition(self.robot_num)
        rvel = self.sim.getAgentVelocity(self.robot_num)
        rrad = self.sim.getAgentRadius(self.robot_num)
        v_pref = self.sim.getAgentMaxSpeed(self.robot_num)
        theta = math.atan2(rvel[1], rvel[0])
        # Robot's state entry. Note that goal is listed as the robot's current
        # position because we aren't using that goal as such, we are just
        # exploring.
        state = [
            rpos[0], rpos[1], rvel[0], rvel[1], rrad, rpos[0], rpos[1],
            v_pref, theta
        ]
        for obs in self.obstacles:
            if(len(obs) > 1):
                # Polygonal obstacle
                o = Polygon(obs)
                p = Point(rpos)
                p1, p2 = nearest_points(o, p)
                state.extend([p1.x, p2.y, 0, 0, self.obs_width])
            else:
                # Point obstacle
                state.extend([obs[0][0], obs[0][1], 0, 0, self.obs_width])
        om = utils.build_occupancy_maps(utils.build_humans(state))
        #??????????????????????????????????????????
        state = utils.transform_and_rotate(state)
        return torch.cat([state, om], dim=-1)

    def reward(self):
        """Calculate the reward the robot receives.

        :return: The (possibly negative) reward of the robot in this time step
            and whether or not the episode is over.
            :rtype: float
            :rtype: boolean

        """
        total = 0
        # Penalties for being close to obstacles:
        #   -0.25 for being within some threshold
        #   -1 for colliding (radii overlap)
        col_reward = -1
        close_reward = -0.25
        close_thresh = 0.2
        robot_pos = self.sim.getAgentPosition(self.robot_num)
        for obs in self.obstacles:
            for vertex in obs:
                if (self.dist(vertex, robot_pos)
                    < self.sim.getAgentRadius(self.robot_num)):
                    # Collision
                    total += col_reward
                elif (self.dist(vertex, robot_pos)
                    < self.sim.getAgentRadius(self.robot_num) + close_thresh):
                    # Closeness penalty
                    total += close_reward
        return total, False

    def dist(self, v1, v2):
        """Calculate the distance between v1 and v2

        :param tuple v1: (x, y) of v1.
        :param tuple v2: (x, y) of v2.
        :return: The Euclidean distance between the two verticies.
            :rtype: float

        """
        return ( (v1[0] - v2[0])**2 + (v1[1] - v2[1])**2 )**0.5

    def build_scene(self, scene):
        """Build up a scene.

        :param string scene: String representing what type of scene to build.
            Pick one from this list:
                `barge_in`,
            If the argument is none of these, a random scene is built
        :return:
            :rtype: None

        """
        self.sim.setTimeStep(0.1)
        if scene == "barge_in":
            num_people = 4
            # Walls
            wall_width = self.obs_width
            wall_length = 2.0
            wall_dist = 1.5
            up_wall_vertices = []
            down_wall_vertices = []
            up_wall_vertices.append((wall_length, 2 * wall_width + wall_dist))
            up_wall_vertices.append((0, 2 * wall_width + wall_dist))
            up_wall_vertices.append((0, wall_dist + wall_width))
            up_wall_vertices.append((wall_length, wall_dist + wall_width))

            down_wall_vertices.append((wall_length, wall_width))
            down_wall_vertices.append((0, wall_width))
            down_wall_vertices.append((0, 0))
            down_wall_vertices.append((wall_length, 0))

            self.sim.addObstacle(up_wall_vertices)
            self.sim.addObstacle(down_wall_vertices)

            self.obstacles.(up_wall_vertices)
            self.obstacles.(down_wall_vertices)

            # "Humans," really just stationary obstacles that fill the corridor
            hums = [
                [(wall_length + 0.2, wall_width + 0.1)],
                [(wall_length + 0.2,
                    wall_width + wall_dist / NUM_PEOPLE + 0.1)],
                [(wall_length + 0.2,
                    wall_width + wall_dist / NUM_PEOPLE * 2 + 0.1)],
                [(wall_length + 0.2,
                    wall_width + wall_dist / NUM_PEOPLE * 3 + 0.1f)]
            ]
            for hum in hums:
                self.sim.addObstacle(hum)
                self.obstacles.append(hum)

            # Add the robot
            self.robot_num = self.sim.addAgent(
                (1.0, -0.15 + wall_width + wall_dist / 2.0),
                1.0, 10, 5.0, 5.0, 0.22, 1.5
            )
        else:       # Build a random scene
            pass

        self.sim.processObstacles()
