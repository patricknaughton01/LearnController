""" simulator.py
Simulator the rl_model uses to train.
RISS 2019

"""

import math
import utils
import torch
import numpy as np
import random
import copy
import learn_general_controller.utils as old_utils
# Used to conveniently find the nearest point on a polygon to the robot
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points
from pyrvo2 import rvo2


class Simulator(object):

    def __init__(self, scene=None, file=None, max_dim=4, reverse=False,
                 single=False, conf=False):
        self.single = single
        self.sim = rvo2.PyRVOSimulator(1.0, 1.0, 10, 5.0, 5.0, 0.22, 1.5)
        self.obstacles = []
        self.robot_num = None
        self.agents = []
        self.goals = []
        self.headings = []
        self.scene = scene
        self.obs_width = 0.0000001
        self.file = file
        self.conf = conf
        self.max_dim = max_dim
        if self.scene is None:
            self.scene = "random"
        self.last_actions = []
        # This should be at least 2 (used for calculating reward)
        self.last_action_capacity = 2
        self.last_action_ind = 0
        # Used to determine if the success controller has failed. If a
        # success controller isn't being used to initialize the scene, this
        # has no effect.
        self.overall_robot_goal = (0, 0)
        self.last_dist = 10**6      # last distance to goal, starts huge
        self.build_scene(self.scene, reverse=reverse)
        self.rot_speed = (self.sim.getAgentMaxSpeed(self.robot_num) *
                          math.pi / 16)
        self.uncertainties = [0.0]
        self.uncertainty_ind = 0

    def do_step(self, action_ind):
        """Run a step of the simulation taking in an action from an
        rl_model.

        :param tensor action_ind: The index of the action to take.
        :return:
            :rtype: None

        """
        action_ind = action_ind.item()
        if len(self.last_actions) < self.last_action_capacity:
            self.last_actions.append(action_ind)
        self.last_actions[self.last_action_ind] = action_ind
        self.last_action_ind = (
                        self.last_action_ind + 1) % self.last_action_capacity
        robot_max_vel = self.sim.getAgentMaxSpeed(self.robot_num)
        # Decode the action selection:
        #   0 => do nothing
        #   1-16 => set velocity to `robot_max_vel/2` at angle
        #       `(action_ind-1) * 2pi/16`
        #   17-32 => velocity to `robot_max_vel` at angle
        #       `(action_ind-17) * 2pi/16`
        #   33-34 => change heading by
        #   else => do nothing
        vel = (0, 0)
        angle = self.headings[self.robot_num]
        if 1 <= action_ind <= 16:
            angle += (action_ind - 1)*(math.pi / 8)
            vel = (
                (robot_max_vel/2) * math.cos(angle),
                (robot_max_vel/2) * math.sin(angle)
            )
        elif 17 <= action_ind <= 32:
            angle += (action_ind - 17)*(math.pi / 8)
            vel = (
                robot_max_vel * math.cos(angle),
                robot_max_vel * math.sin(angle)
            )
        elif action_ind == 33:
            self.headings[self.robot_num] += self.rot_speed
        elif action_ind == 34:
            self.headings[self.robot_num] -= self.rot_speed
        self.headings[self.robot_num] = normalize(self.headings[
                                                      self.robot_num])
        # Set the robot's goal given the action that was selected
        ts = self.sim.getTimeStep()
        pos = self.sim.getAgentPosition(self.robot_num)
        self.goals[self.robot_num] = (
            pos[0] + vel[0] * ts, pos[1] + vel[1] * ts
        )
        self.advance_simulation()

    def forward_simulate(self, success_model, reverse_model, samples=20,
                         max_ts=100, failure_func=None, key="0", conf_val=0.9):
        """Simulates the beginning of the scenario by using the predictions of
        the success_model (a neural network which takes in the state of the
        robot and predicts a mux, muy, sx, sy, and correlation for the
        next state).

        :param torch.nn.Module success_model: a neural network which takes in
            the state of the robot and predicts a mux, muy, sx, sy, and
            correlation for the next state
        :param int samples: The number of samples to take of the network's
            output
        :param int max_ts: The maximum number of timesteps the
            `success_model` is allowed to run before it is cut off
        :param function failure_func: function that returns a boolean value
            and takes in the prediction made by the network. It should
            return True iff the model is making poor predictions/thinks that
            the controller will no longer be successful
        :param string key: An identifier to append to the std_dev filename
        :return: the number of timesteps before the success controller failed
            :rtype: int
        """
        with torch.no_grad():
            # Even though we're evaluating it, we need the model to be in
            # train mode so that dropout layers still work
            success_model.eval()
            pred, h_t = None, None
            r_pred, r_h_t = None, None
            uncertainty = 0.0
            if failure_func is None:
                failure_func = self.base_failure
            i = 0
            #f = open("std_devs_{}.txt".format(key), "w")
            succ_conf = None
            conf = None
            if self.conf:
                conf = open("conf_{}.txt".format(key), "w")
                succ_conf = open("succ_conf_{}.txt".format(key), "w")
            while not failure_func(uncertainty) and i < max_ts:
                mx, my, sx, sy, cov, d_sx, d_sy, d_corr, h_t = \
                    self.get_succ_prediction(
                    success_model, self.success_state(), h_t, samples)
                # uncertainty = (sx+d_sx)**2 + (sy+d_sy)**2
                # f.write("{} {} {} {} {} {}\n".format(sx, sy, rho, d_sx,
                #                                      d_sy, d_corr))
                #print(sx, sy, d_sx, d_sy)
                rpos = self.sim.getAgentPosition(self.robot_num)
                if math.isnan(mx):
                    mx = 0
                if math.isnan(my):
                    my = 0
                self.goals[self.robot_num] = (rpos[0] + mx, rpos[1] + my)
                self.advance_simulation()
                new_r_pos = self.sim.getAgentPosition(self.robot_num)
                successful_move = self.inside_ellipse(
                    rpos, new_r_pos, mx,  my, math.sqrt(sx**2+d_sx**2),
                    math.sqrt(sy**2+d_sy**2),
                    cov + d_corr*d_sx*d_sy, conf_val, conf_file=succ_conf
                )
                mx, my, sx, sy, cov, d_sx, d_sy, d_corr, r_h_t = \
                    self.get_succ_prediction(
                        reverse_model, self.reverse_state(), r_h_t, samples)
                # Compute total uncertainties
                sx *= sx
                sy *= sy
                sx += d_sx**2
                sy += d_sy**2
                cov += d_corr*d_sx*d_sy
                """if (not self.inside_ellipse(new_r_pos, rpos, mx, my, sx, sy,
                                           cov, conf_val, conf)
                    or not successful_move):
                    return -1"""
                i += 1
            if conf is not None:
                conf.close()
                succ_conf.close()
            #print("Finished success simulation after {} steps".format(i))
            return i

    @staticmethod
    def inside_ellipse(start, targ, mx, my, sx, sy, cov, conf_lvl,
                       conf_file=None):
        s = -2.0 * np.log(1-conf_lvl)
        cov_mat = np.matrix([[sx**2, cov],
                             [cov, sy**2]])
        vals, vecs = np.linalg.eig(cov_mat)
        maj, min = vals.max(), vals.min()
        #print(maj, min)
        # a = len of semi-major axis, b = len of semi minor axis
        a, b = np.sqrt(s*maj), np.sqrt(s*min)
        # c is distance from center of ellipse to each focus
        c = np.sqrt(a**2 - b**2)
        maj_vec = vecs[vals.argmax()]
        alpha = np.arctan2(maj_vec[0, 1], maj_vec[0, 0])
        foc_vec = np.array([c*np.cos(alpha), c*np.sin(alpha)])
        center = np.array([start[0] + mx,
                           start[1] + my])
        f1, f2 = center + foc_vec, center - foc_vec
        obs = np.array([targ[0], targ[1]])
        # Outside of ellipse if sum of distances to foci is
        # > 2 a
        d = np.linalg.norm(f1-obs) + np.linalg.norm(f2-obs)
        if conf_file is not None:
            conf_file.write("{} {} {} {} {}\n".format(center[0], center[1],
                                                 a, b, alpha))
        if d > 2*a:
            return True
        return True


    @staticmethod
    def get_succ_prediction(model, state, h_t, samples):
        x = []
        y = []
        sx_list = []
        sy_list = []
        corr_list = []
        model.train()
        next_h_t = None
        model.train()
        #print(state.size())
        with torch.no_grad():
            for i in range(samples):
                pred, next_h_t = model(state, h_t)
                pred = utils.get_coefs(pred.unsqueeze(0))
                x.append(pred[0].item())
                y.append(pred[1].item())
                sx_list.append(pred[2].item())
                sy_list.append(pred[3].item())
                corr_list.append(pred[4].item())
            mx = utils.avg(x)
            my = utils.avg(y)
            sx = utils.std_dev(x)
            sy = utils.std_dev(y)
            cov = utils.cov(x, y)
            data_sx = utils.avg(sx_list)
            data_sy = utils.avg(sy_list)
            data_corr = utils.avg(corr_list)
            return mx, my, sx, sy, cov, data_sx, \
                   data_sy, \
                   data_corr, next_h_t

    def advance_simulation(self):
        """Advance the simulation by moving all the agents towards their
        goals (preferably) and, if `self.file` exists make a record of this
        step.

        :return: None
        """
        ts = self.sim.getTimeStep()
        old_rpos = self.sim.getAgentPosition(self.robot_num)
        # Move all the agents towards their goals
        for agent in self.agents:
            p = self.sim.getAgentPosition(agent)
            g = self.goals[agent]
            vec = ((g[0] - p[0])/ts, (g[1] - p[1])/ts)
            self.sim.setAgentPrefVelocity(agent, vec)
        self.sim.doStep()
        # Check and see if the robot is in collision and reset to old position
        # if so (backtrack on this step)
        # for agent in self.agents:
        #     if (agent != self.robot_num
        #         and utils.dist(self.sim.getAgentPosition(self.robot_num),
        #         self.sim.getAgentPosition(agent)) < (
        #                     self.sim.getAgentRadius(self.robot_num) +
        #                     self.sim.getAgentRadius(agent))):
        #         self.sim.setAgentPosition(self.robot_num, old_rpos)
        if self.file is not None:
            self.update_visualization()

    def base_failure(self, uncertainty):
        """Determines if the predictor has failed. Simply checks the
        uncertainty passed in to determine whether or not it exceeds some
        threshold.

        :param float uncertainty: uncertainty (total uncertainty) in the
            success model
        :return: whether or not the prediction has failed
            :rtype: bool
        """
        return False
        rpos = self.sim.getAgentPosition(self.robot_num)
        gpos = self.overall_robot_goal
        if ((rpos[0]-gpos[0])**2 + (rpos[1]-gpos[1])**2)**0.5 <= \
                self.sim.getAgentRadius(self.robot_num):
            return True
        elif (rpos[0] > gpos[0]):
            return True
        else:
            return False
        self.uncertainties[self.uncertainty_ind] = uncertainty
        self.uncertainty_ind = (self.uncertainty_ind + 1 )%len(
            self.uncertainties)
        # 0.004 is a hard-coded threshold for now that was determined by
        # analyzing multiple barge-in runs.
        return utils.avg(self.uncertainties) > 0.004

    def state(self):
        """Computes the state that will be fed to the DQN. This is composed
        of the robot's rotated and transformed state (see utils.py / ask
        Yash for what that does) and the occupancy map (same advice for how
        this is built).

        :return: Tensor of state values for the robot relative to its
            heading and the occupancy maps for every agent.
            :rtype: Tensor
        """
        state = np.array(self.get_state_arr())
        om = utils.build_occupancy_maps(utils.build_humans(state))
        # We only have a batch of one so just get the first element of
        # transform and rotate
        state = utils.transform_and_rotate(state.reshape((1, -1)))[0]
        return torch.cat((state, om), dim=1)

    def reward(self):
        """Calculates the reward the robot receives.

        :return: The (possibly negative) reward of the robot in this time step
            as a 1x1 tensor and whether or not the episode is over.
            :rtype: tuple
        """
        total = 0
        # Penalties for being close to obstacles:
        #   -0.25 for being within some threshold
        #   -1 for colliding (radii overlap)
        col_reward = -1
        close_reward = -0.25
        close_thresh = 0.2
        pred_mul = 0.05     # Factor to multiply collision penalty by
        ts = self.sim.getTimeStep()
        robot_pos = self.sim.getAgentPosition(self.robot_num)
        r_vel = self.sim.getAgentVelocity(self.robot_num)
        r_rad = self.sim.getAgentRadius(self.robot_num)
        pred_robot_pos = (robot_pos[0] + r_vel[0] * ts,
                          robot_pos[1] + r_vel[1] * ts)
        for agent in self.agents:
            if agent != self.robot_num: # Don't care about self collisions
                # Check for collisions/closeness violations
                a_pos = self.sim.getAgentPosition(agent)
                a_vel = self.sim.getAgentVelocity(agent)
                a_rad = self.sim.getAgentRadius(agent)
                dist = self.dist(a_pos, robot_pos)
                if dist < r_rad + a_rad:
                    total += col_reward
                elif dist < r_rad + a_rad + close_thresh:
                    total += close_reward
                pred_pos = (a_pos[0] + a_vel[0] * ts,
                            a_pos[1] + a_vel[1] * ts)
                # Check for predicted collisions
                if self.dist(pred_robot_pos, pred_pos) < r_rad + a_rad:
                    total += pred_mul * col_reward
                elif self.dist(pred_robot_pos, pred_pos) < r_rad + a_rad + \
                        close_thresh:
                    total += pred_mul * close_reward
                # Check for right hand rule violations (see
                # https://arxiv.org/pdf/1703.08862.pdf for more details)
                # Note the precise sizes of all the zones are determined by
                # the radius of the robot.
                a_ang = math.atan2(a_pos[1] - robot_pos[1], a_pos[0] -
                                   robot_pos[0])
                r_ang = math.atan2(r_vel[1], r_vel[0])
                a_rel_ang = a_ang - r_ang
                a_vel_ang = math.atan2(a_vel[1], a_vel[0])
                vel_ang_diff = a_vel_ang - r_ang
                # Transformed coordinates where x-axis is aligned along
                # robot's velocity vector
                x_rel = dist * math.cos(a_rel_ang)
                y_rel = dist * math.sin(a_rel_ang)
                ovtk_min_x = 0
                ovtk_max_x = 3 * r_rad
                ovtk_min_y = 0
                ovtk_max_y = 2 * r_rad
                pass_min_x = 1.5 * r_rad
                pass_max_x = 4 * r_rad
                pass_min_y = -3 * r_rad
                pass_max_y = 0
                cross_min_ang = -math.pi/4.0
                cross_max_ang = math.pi/4.0
                cross_max_dist = 4 * r_rad
                # Penalty for violating one of these rules. Same weight for
                # all rules right now
                # Can only incur a social penalty if the robot is moving
                if utils.dist(r_vel, (0.0, 0.0)) > 0.0:
                    social_reward = 0.0
                else:
                    social_reward = 0.0
                # Check both relative position of the other agent as well as
                # the angle between the agent's and robot's velocities
                if ovtk_min_x <= x_rel <= ovtk_max_x and ovtk_min_y <= \
                        y_rel <= ovtk_max_y and math.fabs(vel_ang_diff) < \
                        math.pi/4.0:
                    total += social_reward
                if pass_min_x <= x_rel <= pass_max_x and pass_min_y <= y_rel \
                        <= pass_max_y and math.fabs(vel_ang_diff) > \
                        3*math.pi/4.0:
                    total += social_reward
                if -3*math.pi/4.0 <= vel_ang_diff <= -math.pi/4.0 and dist <\
                        cross_max_dist and cross_min_ang <= a_rel_ang <= \
                        cross_max_ang:
                    total += social_reward
        for obs in self.obstacles:
            dist = 0
            if len(obs) > 1:
                # Polygonal obstacle
                o = Polygon(obs)
                p = Point(robot_pos)
                p1, p2 = nearest_points(o, p)
                if not o.contains(p):
                    dist = self.dist((p1.x, p1.y), (p2.x, p2.y))
            else:
                # Point obstacle
                dist = self.dist(robot_pos, obs[0])
            if dist < self.sim.getAgentRadius(self.robot_num):
                total += col_reward
            elif (dist < self.sim.getAgentRadius(self.robot_num)
                  + close_thresh):
                total += close_reward
        prev_action_ind = self.last_action_ind - 1
        prev2_action_ind = self.last_action_ind - 2
        while prev_action_ind < 0:
            prev_action_ind += self.last_action_capacity
        while prev2_action_ind < 0:
            prev2_action_ind += self.last_action_capacity
        prev_action_ind %= self.last_action_capacity
        prev2_action_ind %= self.last_action_capacity
        # Penalize non-still actions
        if len(self.last_actions) > 0 and self.last_actions[prev_action_ind] \
                != 0:
            total += -0.01
        # Encourage smooth trajectories by penalizing changing actions,
        # except for starting up if the robot was previously stopped
        if len(self.last_actions) > 1 and self.last_actions[prev_action_ind] \
                != self.last_actions[prev2_action_ind] and \
                self.last_actions[prev2_action_ind] != 0:
            total += -0.01
        return torch.tensor([[total]], dtype=torch.float), False

    @staticmethod
    def dist(v1, v2):
        """Calculate the distance between v1 and v2

        :param tuple v1: (x, y) of v1.
        :param tuple v2: (x, y) of v2.
        :return: The Euclidean distance between the two verticies.
            :rtype: float

        """
        return ( (v1[0] - v2[0])**2 + (v1[1] - v2[1])**2 )**0.5

    def update_visualization(self):
        """Adds the current state of the robot to this simulator's file for
        viewing later.

        :return: String that was written to the file
            :rtype: str
        """
        rpos = self.sim.getAgentPosition(self.robot_num)
        rvel = self.sim.getAgentVelocity(self.robot_num)
        rrad = self.sim.getAgentRadius(self.robot_num)
        v_pref = self.sim.getAgentMaxSpeed(self.robot_num)
        theta = math.atan2(rvel[1], rvel[0])
        return_str = ""
        return_str += str(self.sim.getGlobalTime()) + " "
        return_str += "(" + str(rpos[0]) + "," + str(rpos[1]) + ") "
        return_str += "(" + str(rvel[0]) + "," + str(rvel[1]) + ") "
        return_str += str(rrad) + " "
        return_str += str(self.headings[self.robot_num]) + " "
        return_str += ("(" + str(self.overall_robot_goal[0])
                       + "," + str(self.overall_robot_goal[1]) + ") ")
        return_str += str(v_pref) + " "
        return_str += str(theta) + " "
        if not self.single:
            for agent in self.agents:
                if agent != self.robot_num: # We already wrote out the robot
                    pos = self.sim.getAgentPosition(agent)
                    vel = self.sim.getAgentVelocity(agent)
                    rad = self.sim.getAgentRadius(agent)
                    return_str += "(" + str(pos[0]) + "," + str(pos[1]) + ") "
                    return_str += "(" + str(vel[0]) + "," + str(vel[1]) + ") "
                    return_str += str(rad) + " "
                    return_str += str(self.headings[agent]) + " "
        for obs in self.obstacles:
            if len(obs) > 1:
                # Polygonal obstacle
                o = Polygon(obs)
                p = Point(rpos)
                p1, p2 = nearest_points(o, p)
                return_str += "(" + str(p1.x) + "," + str(p1.y) + ") "
                return_str += "(0,0) "
                return_str += str(self.obs_width) + " "
                return_str += str(self.headings[self.robot_num]) + " "
            else:
                # Point obstacle
                return_str += \
                    "(" + str(obs[0][0]) + "," +str(obs[0][1]) + ") "
                return_str += "(0,0) "
                return_str += str(self.obs_width) + " "
                return_str += str(self.headings[self.robot_num]) + " "
        return_str += "\n"
        if self.file is not None:
            self.file.write(return_str)
        return return_str

    def get_state_arr(self):
        """Calculate the state of the robot and all the obstacles in the scene.
        Takes the form of
        [
            robot pos, robot vel, robot rad, robot goal, robot heading, robot
            v_pref, robot theta,
            obs1 position, obs1 vel, obs1 rad, obs1 heading,
            obs2...
            ...
        ]
        Note that none of these are tuples, if the value is a vector (like
        position) its elements are just listed out.

        :return: Array with the state of all the obstacles in the scene plus
            the robot.
            :rtype: list
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
            rpos[0], rpos[1], rvel[0], rvel[1], rrad,
            self.headings[self.robot_num], rpos[0], rpos[1],
            v_pref, theta
        ]
        for agent in self.agents:
            if agent != self.robot_num: # We already accounted for the robot
                pos = self.sim.getAgentPosition(agent)
                vel = self.sim.getAgentVelocity(agent)
                rad = self.sim.getAgentRadius(agent)
                state.extend([pos[0], pos[1], vel[0], vel[1], rad,
                              self.headings[agent]])
        for obs in self.obstacles:
            if len(obs) > 1:
                # Polygonal obstacle
                o = Polygon(obs)
                p = Point(rpos)
                p1, p2 = nearest_points(o, p)
                # Velocity is always 0 for obstacles
                # Heading is same as robot's
                state.extend([p1.x, p2.y, 0, 0, self.obs_width,
                              self.headings[self.robot_num]])
            else:
                # Point obstacle
                state.extend([obs[0][0], obs[0][1], 0, 0, self.obs_width,
                              self.headings[self.robot_num]])
        return state

    def success_state(self):
        """Computes the state that will be fed to the success model. This is
        composed of the robot's rotated and transformed state (see utils.py
        / ask Yash for what that does) and the occupancy map (same advice
        for how this is built).

        :return: Tensor of state values for the robot relative to its
            velocity and the occupancy maps for every agent.
            :rtype: Tensor
        """
        state = np.array(self.get_success_state_arr())
        om = utils.build_occupancy_maps(utils.build_humans(state))
        #print("OM: ", om.size())
        # We only have a batch of one so just get the first element of
        # transform and rotate
        state = utils.transform_and_rotate(state.reshape((1, -1)))[0]
        #print("State: ", state.size())
        return torch.cat((state, om), dim=1).unsqueeze(0)

    def get_success_state_arr(self):
        """Calculate the state of the robot and all the obstacles in the
        scene such that the success controller can interpret it.
        Takes the form of
        [
            robot pos, robot vel, robot rad, robot goal, robot v_pref,
            robot theta,
            obs1 position, obs1 vel, obs1 rad,
            obs2...
            ...
        ]
        Note that none of these are tuples, if the value is a vector (like
        position) its elements are just listed out.

        :return: Array with the state of all the obstacles in the scene plus
            the robot.
            :rtype: list
        """
        rpos = self.sim.getAgentPosition(self.robot_num)
        rvel = self.sim.getAgentVelocity(self.robot_num)
        rrad = self.sim.getAgentRadius(self.robot_num)
        v_pref = self.sim.getAgentMaxSpeed(self.robot_num)
        theta = math.atan2(rvel[1], rvel[0])
        # Robot's state entry.
        state = [
            rpos[0], rpos[1], rvel[0], rvel[1], rrad,
            self.headings[self.robot_num], self.overall_robot_goal[0],
            self.overall_robot_goal[1], v_pref, theta
        ]
        for agent in self.agents:
            if agent != self.robot_num:  # We already accounted for the robot
                pos = self.sim.getAgentPosition(agent)
                vel = self.sim.getAgentVelocity(agent)
                rad = self.sim.getAgentRadius(agent)
                state.extend([pos[0], pos[1], vel[0], vel[1], rad,
                              self.headings[agent]])
        for obs in self.obstacles:
            if len(obs) > 1:
                # Polygonal obstacle
                o = Polygon(obs)
                p = Point(rpos)
                p1, p2 = nearest_points(o, p)
                # Velocity is always 0 for obstacles
                # Heading is same as robot's
                state.extend([p1.x, p2.y, 0, 0, self.obs_width,
                              self.headings[self.robot_num]])
            else:
                # Point obstacle
                state.extend([obs[0][0], obs[0][1], 0, 0, self.obs_width,
                              self.headings[self.robot_num]])
        return state

    def reverse_state(self):
        """Computes the state that will be fed to the success model. This is
        composed of the robot's rotated and transformed state (see utils.py
        / ask Yash for what that does) and the occupancy map (same advice
        for how this is built).

        :return: Tensor of state values for the robot relative to its
            velocity and the occupancy maps for every agent.
            :rtype: Tensor
        """
        state = np.array(self.get_reverse_state_arr())
        om = utils.build_occupancy_maps(utils.build_humans(state))
        # print("OM: ", om.size())
        # We only have a batch of one so just get the first element of
        # transform and rotate
        state = utils.transform_and_rotate(state.reshape((1, -1)))[0]
        # print("State: ", state.size())
        return torch.cat((state, om), dim=1).unsqueeze(0)

    def get_reverse_state_arr(self):
        """Calculate the state of the robot and all the obstacles in the
        scene such that the success controller can interpret it.
        Takes the form of
        [
            robot pos, robot vel, robot rad, robot goal, robot v_pref,
            robot theta,
            obs1 position, obs1 vel, obs1 rad,
            obs2...
            ...
        ]
        Note that none of these are tuples, if the value is a vector (like
        position) its elements are just listed out.

        :return: Array with the state of all the obstacles in the scene plus
            the robot.
            :rtype: list
        """
        rpos = self.sim.getAgentPosition(self.robot_num)
        rvel = self.sim.getAgentVelocity(self.robot_num)
        rrad = self.sim.getAgentRadius(self.robot_num)
        v_pref = self.sim.getAgentMaxSpeed(self.robot_num)
        theta = math.atan2(rvel[1], rvel[0])
        # Robot's state entry.
        state = [
            rpos[0], rpos[1], rvel[0], rvel[1], rrad,
            self.headings[self.robot_num], self.overall_robot_goal[0],
            self.overall_robot_goal[1], v_pref, theta
        ]
        for obs in self.obstacles:
            if len(obs) > 1:
                # Polygonal obstacle
                o = Polygon(obs)
                p = Point(rpos)
                p1, p2 = nearest_points(o, p)
                # Velocity is always 0 for obstacles
                # Heading is same as robot's
                state.extend([p1.x, p2.y, 0, 0, self.obs_width,
                              self.headings[self.robot_num]])
            else:
                # Point obstacle
                state.extend([obs[0][0], obs[0][1], 0, 0, self.obs_width,
                              self.headings[self.robot_num]])
        return state

    def build_scene(self, scene, reverse=False):
        """Build up a scene.

        :param string scene: String representing what type of scene to build.
            Pick one from this list:
                `dynamic_barge_in<_left|_top|_bottom>`,
                `barge_in<_left|_top|_bottom>`,
                `crossing`,
                `overtake`,
            If the argument is none of these, a random scene is built
        :return:
            :rtype: None

        """
        success_trial = 'success' in scene
        if (scene.startswith("barge_in")
                or scene.startswith("dynamic_barge_in")):
            num_people = 4
            # Walls
            wall_width = 1.0
            wall_length = 7.0
            wall_dist = 4.0
            human_goal_dist = 3.0
            up_wall_vertices = [
                (wall_length, 2*wall_width + wall_dist),
                (0, 2*wall_width + wall_dist),
                (0, wall_width + wall_dist),
                (wall_length, wall_width + wall_dist)
            ]
            down_wall_vertices = [
                (wall_length, wall_width),
                (0, wall_width),
                (0, 0),
                (wall_length, 0)
            ]
            self.obstacles.append(up_wall_vertices)
            self.obstacles.append(down_wall_vertices)

            # Add the robot
            robot_pos = (wall_length - 1.0, wall_width + wall_dist/2.0 +
                         randomize(-0.5, 0.5))
            self.overall_robot_goal = (wall_length + 3.0, wall_width +
                                       wall_dist/2.0 + randomize(-0.5, 0.5))
            if reverse:
                tmp = robot_pos
                robot_pos = self.overall_robot_goal
                self.overall_robot_goal = tmp
            self.robot_num = self.sim.addAgent(
                robot_pos,
                10.0, 10, 2.0, 5.0, 0.5, 3.0, (0, 0)
            )
            self.agents.append(self.robot_num)
            self.goals.append(robot_pos)
            self.headings.append(randomize(-math.pi/8, math.pi/8))

            hum_perb = 0.1  # Random perturbation to add to human positions
            if scene.startswith("barge_in"):
                # "Humans," really just obstacles that fill the corridor
                # Note that they are just the same vertex thrice because RVO2
                # didn't like one vert obstacles and shapely needs 3 verticies
                # to treat them like a polygon (used to find dist from robot
                # to obstacles).
                hums = [
                    [
                        (wall_length + 0.2, wall_width + 0.1),
                        (wall_length + 0.2, wall_width + 0.1 + 0.5),
                        (wall_length + 0.2 + 0.1, wall_width + 0.1)
                    ],
                    [
                        (wall_length + 0.2,
                         wall_width + wall_dist / num_people + 0.1),
                        (wall_length + 0.2,
                         wall_width + wall_dist / num_people + 0.1 + 0.5),
                        (wall_length + 0.2 + 0.1,
                         wall_width + wall_dist / num_people + 0.1)
                    ],
                    [
                        (wall_length + 0.2,
                         wall_width + wall_dist / num_people * 2 + 0.1),
                        (wall_length + 0.2,
                         wall_width + wall_dist / num_people * 2 + 0.1 + 0.5),
                        (wall_length + 0.2 + 0.1,
                         wall_width + wall_dist / num_people * 2 + 0.1)
                    ],
                    [
                        (wall_length + 0.2,
                         wall_width + wall_dist / num_people * 3 + 0.1),
                        (wall_length + 0.2,
                         wall_width + wall_dist / num_people * 3 + 0.1 + 0.5),
                        (wall_length + 0.2 + 0.1,
                         wall_width + wall_dist / num_people * 3 + 0.1)
                    ]
                ]
                for hum in hums:
                    for i, vert in enumerate(hum):
                        hum[i] = (vert[0] + hum_perb * random.random(),
                                  vert[1] + hum_perb * random.random())
                    self.obstacles.append(hum)
            else:
                if success_trial:
                    num_people = 4
                    pos1 = (wall_length + randomize(0, 0.5),
                        wall_width + (wall_dist / num_people) / 2.0)
                    goal1 = (pos1[0] + human_goal_dist +
                             randomize(-0.2, 0.2), pos1[1] - 1.0)

                    pos2 = (wall_length + randomize(1.0, 1.5), pos1[1] + 1.0)
                    goal2 = (pos2[0] + human_goal_dist +
                             randomize(-0.2, 0.2), pos1[1] - 0.5)

                    pos3 = (wall_length + randomize(0, 0.5),
                            pos2[1] + 1.0)
                    goal3 = (pos3[0] + human_goal_dist +
                             randomize(-0.2, 0.2), pos3[1] + 0.5)

                    pos4 = (wall_length + randomize(1.0, 1.5), pos3[1] + 1.0)
                    goal4 = (pos4[0] + human_goal_dist +
                             randomize(-0.2, 0.2), pos4[1] + 1.0)

                    poses = [pos1, pos2, pos3, pos4]
                    gs = [goal1, goal2, goal3, goal4]
                    if reverse:
                        poses = [goal1, goal2, goal3, goal4]
                        gs = [pos1, pos2, pos3, pos4]
                    for p in poses:
                        self.agents.append(self.sim.addAgent(
                            p, 10.0, 10, 2.0, 5.0,
                            0.5, 0.7, (0, 0)
                        ))
                        self.headings.append(randomize(-math.pi/8, math.pi/8))
                    for g in gs:
                        self.goals.append(g)
                else:
                    # Make humans actual agents that move either towards or
                    # away from the robot
                    min_hum = 4
                    max_hum = 4
                    max_hum_rad = 0.5
                    num_hum = random.randint(min_hum, max_hum)
                    for i in range(num_hum):
                        # Stack humans in front of the passage
                        pos = (
                            wall_length+2*max_hum_rad
                                + random.random() * hum_perb,
                            wall_width+wall_dist+0.1
                                + random.random() * hum_perb
                                - 2*(max_hum_rad + hum_perb)
                                * (max_hum/num_hum) * i
                        )
                        self.agents.append(self.sim.addAgent(
                            pos, 10.0, 10, 2.0, 5.0, 0.5,
                            0.7, (0, 0)
                        ))
                        goal_min = -2.0
                        goal_max = -1.0
                        self.goals.append((
                            pos[0] + randomize(goal_min, goal_max),
                            wall_width + wall_dist/2.0
                        ))
                        self.headings.append(
                            normalize(randomize(7*math.pi/8, 9*math.pi/8))
                        )
            # By default, builds a scene in which the robot barges in to the
            # right. If one of the following specific scenes is provided,
            if scene.endswith("left"):    # Negate x coordinate
                for obs in self.obstacles:
                    for i, vert in enumerate(obs):
                        obs[i] = (-vert[0], vert[1])
                    obs.reverse()  # Verticies must be in ccw order
                for agent in self.agents:
                    pos = self.sim.getAgentPosition(agent)
                    self.sim.setAgentPosition(agent, (-pos[0], pos[1]))
                for i, goal in enumerate(self.goals):
                    self.goals[i] = (-goal[0], goal[1])
                for i, heading in enumerate(self.headings):
                    self.headings[i] = normalize(heading + math.pi)
                self.overall_robot_goal = (-self.overall_robot_goal[0],
                                           self.overall_robot_goal[1])
            elif scene.endswith("top"):   # flip x and y coordinates
                for obs in self.obstacles:
                    for i, vert in enumerate(obs):
                        obs[i] = (vert[1], vert[0])
                    obs.reverse()     # Verticies must be in ccw order
                for agent in self.agents:
                    pos = self.sim.getAgentPosition(agent)
                    self.sim.setAgentPosition(agent, (pos[1], pos[0]))
                for i, goal in enumerate(self.goals):
                    self.goals[i] = (goal[1], goal[0])
                for i, heading in enumerate(self.headings):
                    self.headings[i] = normalize(heading + math.pi/2)
                self.overall_robot_goal = (self.overall_robot_goal[1],
                                           self.overall_robot_goal[0])
            elif scene.endswith("bottom"):
                # flip x and y coordinates
                # then negate new y
                for obs in self.obstacles:
                    for i, vert in enumerate(obs):
                        obs[i] = (vert[1], -vert[0])
                for agent in self.agents:
                    pos = self.sim.getAgentPosition(agent)
                    self.sim.setAgentPosition(agent, (pos[1], -pos[0]))
                for i, goal in enumerate(self.goals):
                    self.goals[i] = (goal[1], -goal[0])
                for i, heading in enumerate(self.headings):
                    self.headings[i] = normalize(heading - math.pi/2)
                self.overall_robot_goal = (self.overall_robot_goal[1],
                                           -self.overall_robot_goal[0])
            for obs in self.obstacles:
                self.sim.addObstacle(obs)
        elif scene == "crossing":   # Build crossing scene
            position1 = (-1.5, 25.0)
            position2 = (2.5, 25.0)
            self.robot_num = self.sim.addAgent(
                position1, 15.0, 10, 5.0, 5.0,
                randomize(0.15, 0.25), randomize(0.8, 2.0)
            )
            self.agents.append(self.robot_num)
            self.goals.append(position2)
            self.headings.append(normalize(randomize(-math.pi/8, math.pi/8)))

            self.agents.append(
                self.sim.addAgent(
                    position2, 15.0, 10, 5.0, 5.0, randomize(0.15, 0.25),
                    randomize(0.8, 2.0)
                )
            )
            self.goals.append(position1)
            self.headings.append(normalize(randomize(7 * math.pi/8,
                                                     9 * math.pi/8)))
        elif scene.startswith("overtake"):     # overtaking scene
            neighbor_dist = 10.0
            max_neighbors = 10
            time_horizon = 2.0
            time_horizon_obst = 5.0
            radius = 0.3
            robot_max_speed = 3.0
            slow_human_max_speed = 0.4
            human_max_speed = 0.7

            pos1 = (randomize(-2.0, 1.5), randomize(-1.0, 1.0))  # Robot
            # Human to overtake
            pos2 = (randomize(-1.0, -0.5), randomize(-1.0, 1.0))
            hum_goal = (randomize(5.0, 6.0), randomize(-1.0, 1.0))
            # Robot
            self.robot_num = self.sim.addAgent(pos1, neighbor_dist,
                max_neighbors, time_horizon, time_horizon_obst,
                radius, robot_max_speed, (0, 0))
            self.goals.append(pos1)  # Robot has no explicit goal at first
            # Used to determine if success controller has failed.
            self.overall_robot_goal = hum_goal
            self.agents.append(self.robot_num)
            self.headings.append(
                normalize(randomize(-math.pi / 8, math.pi / 8)))
            # Human to overtake
            self.agents.append(self.sim.addAgent(pos2, neighbor_dist,
                max_neighbors, time_horizon, time_horizon_obst,
                radius, slow_human_max_speed, (0, 0)))
            self.goals.append(hum_goal)
            self.headings.append(
                normalize(randomize(-math.pi / 8, math.pi / 8)))
            # Another human going the opposite way
            self.agents.append(self.sim.addAgent(hum_goal, neighbor_dist,
                max_neighbors, time_horizon, time_horizon_obst,
                radius, human_max_speed, (0, 0)))
            self.goals.append(pos2)
            self.headings.append(
                normalize(math.pi + randomize(-math.pi / 8,
                                                      math.pi / 8)))
            if not success_trial:
                # Add other humans walking around in the middle of the path...
                self.agents.append(self.sim.addAgent(
                    (randomize(1.0, 2.0), randomize(-1.0, -2.0)), 15.0, 10, 5.0,
                    5.0, randomize(0.15, 0.25), randomize(1.5, 2.0), (0, 0)))
                self.goals.append((randomize(-1.0, 0.0), randomize(0.0, 1.0)))
                self.headings.append(
                    normalize(3 * math.pi / 4 + randomize(-math.pi / 8,
                                                          math.pi / 8)))
                self.agents.append(self.sim.addAgent(
                    (randomize(0.0, 1.0), randomize(0.0, -1.0)), 15.0, 10, 5.0,
                    5.0, randomize(0.15, 0.25), randomize(1.5, 2.0), (0, 0)))
                self.goals.append((randomize(-2.0, -1.0), randomize(1.0, 2.0)))
                self.headings.append(
                    normalize(3 * math.pi / 4 + randomize(-math.pi / 8,
                                                          math.pi / 8)))
                self.agents.append(self.sim.addAgent(
                    (randomize(-2.0, -1.0), randomize(1.0, 2.0)), 15.0, 10, 5.0,
                    5.0, randomize(0.15, 0.25), randomize(1.5, 2.0), (0, 0)))
                self.goals.append((randomize(1.0, 2.0), randomize(-2.0, -1.0)))
                self.headings.append(
                    normalize(-math.pi / 4 + randomize(-math.pi / 8,
                                                          math.pi / 8)))
                self.agents.append(self.sim.addAgent(
                    (randomize(0.0, -1.0), randomize(0.0, 1.0)), 15.0, 10, 5.0,
                    5.0, randomize(0.15, 0.25), randomize(1.5, 2.0), (0, 0)))
                self.goals.append((randomize(0.0, 1.0), randomize(0.0, -1.0)))
                self.headings.append(
                    normalize(-math.pi / 4 + randomize(-math.pi / 8,
                                                       math.pi / 8)))
            """else:
                self.agents.append(self.sim.addAgent(
                    (hum_goal[0] + randomize(0.5, 0.7),
                     hum_goal[1] + randomize(0.5, 0.7)), 15.0, 10, 5.0, 5.0,
                    randomize(0.15, 0.25), randomize(1.5, 2.0), (0, 0)
                ))
                self.goals.append((pos1[0] + randomize(0.5, 0.7), pos1[1] +
                                   randomize(0.5, 0.7)))
                self.headings.append(normalize(randomize(-math.pi, math.pi)))

                self.agents.append(self.sim.addAgent(
                    (pos1[0] + randomize(0.5, 0.7),
                     pos1[1] + randomize(0.5, 0.7)), 15.0, 10, 5.0, 5.0,
                    randomize(0.15, 0.25), randomize(1.5, 2.0), (0, 0)
                ))
                self.goals.append((hum_goal[0] + randomize(0.5, 0.7),
                                   hum_goal[1] + randomize(0.5, 0.7)))
                self.headings.append(normalize(randomize(-math.pi, math.pi)))"""

        else:       # Build a random scene
            max_dim = self.max_dim    # Maximum x and y start/goal locations
            min_agents = 5
            max_agents = 10
            min_obs = 5
            max_obs = 10
            num_agents = random.randint(min_agents, max_agents)
            num_obstacles = random.randint(min_obs, max_obs)
            # Create the robot
            robot_pos = (max_dim * random.random(), max_dim * random.random())
            self.robot_num = self.sim.addAgent(
                robot_pos
            )
            self.agents.append(self.robot_num)
            self.goals.append(robot_pos)
            self.headings.append(normalize(randomize(-math.pi, math.pi)))
            # For this, just create small square obstacles
            for i in range(num_obstacles):
                pt = (max_dim * random.random(), max_dim * random.random())
                width = 0.2
                o = [
                    pt, (pt[0] + width, pt[1]), (pt[0] + width, pt[1] + width),
                    (pt[0], pt[1] + width)
                ]
                self.obstacles.append(o)
                self.sim.addObstacle(o)
            # Create agents in random spots with random goals
            for i in range(num_agents):
                self.agents.append(
                    self.sim.addAgent(
                        (max_dim * random.random(), max_dim * random.random())
                    )
                )
                self.goals.append(
                    (max_dim * random.random(), max_dim * random.random())
                )
                self.headings.append(normalize(randomize(-math.pi, math.pi)))

        self.sim.processObstacles()
        if self.file is not None:
            # First line is obstacles in the scene
            self.file.write(str(self.obstacles) + "\n")
            self.file.write("timestamp position0 velocity0 radius0 "
                            "heading0 goal ")
            self.file.write("pref_speed theta ")
            num = 1
            for _ in range(len(self.agents) - 1):
                self.file.write("position" + str(num) + " ")
                self.file.write("velocity" + str(num) + " ")
                self.file.write("radius" + str(num) + " ")
                self.file.write("heading" + str(num) + " ")
                num += 1
            for _ in self.obstacles:
                self.file.write("position" + str(num) + " ")
                self.file.write("velocity" + str(num) + " ")
                self.file.write("radius" + str(num) + " ")
                self.file.write("heading" + str(num) + " ")
                num += 1
            self.file.write("\n")
            self.update_visualization()

    def test_reward(self):
        """Tests the reward function to make sure it evaluates correctly

        :return: whether the reward function is correct
            :rtype: bool
        """
        success = True
        old_sim = self.sim
        old_robot_num = self.robot_num
        old_agents = copy.deepcopy(self.agents)
        old_obstacles = copy.deepcopy(self.obstacles)
        old_goals = copy.deepcopy(self.goals)
        old_action_list = copy.deepcopy(self.last_actions)

        # Test collision penalties and overtaking penalty
        self.sim = rvo2.PyRVOSimulator(
            0.1, 1.0, 10, 5.0, 5.0, 0.2, 1.5, (0,0)
        )
        self.obstacles = []
        self.goals = []
        self.last_actions = []
        self.robot_num = self.sim.addAgent((0, 0))
        self.agents = [self.robot_num]
        self.agents.append(self.sim.addAgent((0.1, 0.1)))
        self.agents.append(self.sim.addAgent((-0.1, 0.1)))
        self.agents.append(self.sim.addAgent((0.1, -0.1)))
        self.agents.append(self.sim.addAgent((-0.1, -0.1)))
        r = self.reward()[0].item()
        exp = -4.22
        if r != exp:
            success = False
        print("Actual reward: ", r, "Expected: ", exp)
        print("Explanation: -4 for 4 collisions, -0.2 for 4 predicted "
              "collisions, -0.02 for overtake penalty with top right agent")

        # Test closeness penalties and overtaking penalty
        self.agents = []
        self.sim = rvo2.PyRVOSimulator(
            0.1, 1.0, 10, 5.0, 5.0, 0.2, 1.5, (0,0)
        )
        self.robot_num = self.sim.addAgent((0, 0))
        self.agents = [self.robot_num]
        self.agents.append(self.sim.addAgent((0.35, 0.35)))
        self.agents.append(self.sim.addAgent((0.35, -0.35)))
        self.agents.append(self.sim.addAgent((-0.35, 0.35)))
        self.agents.append(self.sim.addAgent((-0.35, -0.35)))
        r = self.reward()[0].item()
        exp = -1.02
        if r != exp:
            success = False
        print("Actual reward: ", r, "Expected: ", exp)
        print("Explanation: -1 for 4 closeness violations, -0.02 for "
              "overtake penalty with top right agent")

        # Test passing penalty
        self.agents = []
        self.sim = rvo2.PyRVOSimulator(
            0.1, 1.0, 10, 5.0, 5.0, 0.2, 1.5, (0, 0)
        )
        self.robot_num = self.sim.addAgent((0, 0))
        self.agents = [self.robot_num]
        self.agents.append(self.sim.addAgent((0.7, -0.5), 1.0, 10, 5.0, 5.0,
                                             0.2, 1.5, (-0.5, 0)))
        r = self.reward()[0].item()
        exp = -0.02
        if r != exp:
            success = False
        print("Actual reward: ", r, "Expected: ", exp)
        print("Explanation: -0.02 for passing violation")

        # Test crossing penalty
        self.agents = []
        self.sim = rvo2.PyRVOSimulator(
            0.1, 1.0, 10, 5.0, 5.0, 0.2, 1.5, (0, 0)
        )
        self.robot_num = self.sim.addAgent((0, 0))
        self.agents = [self.robot_num]
        self.agents.append(self.sim.addAgent((0.35, 0.3), 1.0, 10, 5.0, 5.0,
                                             0.2, 1.5, (0, -0.5)))
        r = self.reward()[0].item()
        exp = -0.27
        if r != exp:
            success = False
        print("Actual reward: ", r, "Expected: ", exp)
        print("Explanation: -0.02 for crossing violation, -0.25 for "
              "closeness violation")

        # Test action penalty (moving)
        self.agents = []
        self.sim = rvo2.PyRVOSimulator(
            0.1, 1.0, 10, 5.0, 5.0, 0.2, 1.5, (0, 0)
        )
        self.robot_num = self.sim.addAgent((0, 0))
        self.last_actions = [1, 1]
        self.last_action_ind = 0
        r = self.reward()[0].item()
        exp = -0.01
        if r != exp:
            success = False
        print("Actual reward: ", r, "Expected: ", exp)
        print("Explanation: -0.01 for moving")

        # Test action penalty (changing actions)
        self.agents = []
        self.sim = rvo2.PyRVOSimulator(
            0.1, 1.0, 10, 5.0, 5.0, 0.2, 1.5, (0, 0)
        )
        self.robot_num = self.sim.addAgent((0, 0))
        self.last_actions = [1, 0]
        self.last_action_ind = 0
        r = self.reward()[0].item()
        exp = -0.01
        if r != exp:
            success = False
        print("Actual reward: ", r, "Expected: ", exp)
        print("Explanation: -0.01 for changing actions")

        self.sim = old_sim
        self.robot_num = old_robot_num
        self.agents = old_agents
        self.obstacles = old_obstacles
        self.goals = old_goals
        self.last_actions = old_action_list
        return success


def randomize(lower, upper):
    """Generate a random float in the range [min, max)

    :param float lower: The minimum number of the range
    :param float upper: The maximum number of the range
    :return: A random float in the range [min, max)
        :rtype: float
    """
    return lower + (random.random() * (upper - lower))


def normalize(angle):
    """Normalize the given angle so that it is in [-pi, pi]

    :param float angle: Angle to normalize in radians.
    :return: Noramlized angle
        :rtype: float
    """
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


if __name__ == "__main__":
    sim = Simulator()
    sim.test_reward()
    print(sim.state())
