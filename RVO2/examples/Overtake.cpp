/*
 * Overtake.cpp
 * RVO2 Library
 *
 * Copyright 2008 University of North Carolina at Chapel Hill
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please send all bug reports to <geom@cs.unc.edu>.
 *
 * The authors may be contacted via:
 *
 * Jur van den Berg, Stephen J. Guy, Jamie Snape, Ming C. Lin, Dinesh Manocha
 * Dept. of Computer Science
 * 201 S. Columbia St.
 * Frederick P. Brooks, Jr. Computer Science Bldg.
 * Chapel Hill, N.C. 27599-3175
 * United States of America
 *
 * <http://gamma.cs.unc.edu/RVO2/>
 */

/*
 * Example file showing a robot overtake a human in a scene with other humans.
 */

#include <cmath>
#include <math.h>
#include <cstdlib>
#include <random>

#include <vector>

#include <iostream>
#include <fstream>

#include <ctime>

#if _OPENMP
#include <omp.h>
#endif

#include <RVO.h>

#ifndef M_PI
const float M_PI = 3.14159265358979323846f;
#endif

#define START_EXP 50
#define MAX_NUM_EXPS 10

/* Store the goals of the agents. */
std::vector<RVO::Vector2> goals;

float randomize(float LO=0.0f, float HI=1.0f)
{
    return LO + static_cast <float> (rand()) / static_cast <float> (RAND_MAX / (HI - LO));
}

std::vector<std::vector<RVO::Vector2> > setupScenario(RVO::RVOSimulator *sim)
{
	// std::srand(static_cast<unsigned int>(std::time(NULL)));

	/* Specify the global time step of the simulation. */
	//sim->setTimeStep(0.25f);
	sim->setTimeStep(0.1f);

	/*
	 * Add agents, specifying their start position, and store their goals on the
	 * opposite side of the environment.
	 */

    RVO::Vector2 robot_pos(randomize(-2.0f, -1.5f), randomize(-2.0f, -1.5f));
    RVO::Vector2 robot_goal(randomize(2.0f, 3.0f), randomize(2.0f, 3.0f));

    sim->addAgent(robot_pos, 15.0f, 10, 5.0f, 5.0f, randomize(0.15f, 0.22f),
    randomize(1.5f, 2.0f));
    goals.push_back(robot_goal);

    // Human to overtake
    RVO::Vector2 human_pos1(randomize(-1.0f, -0.5f), randomize(-1.0f, -0.5f));
    RVO::Vector2 human_goal1(robot_goal);

    // Another human going opposite way
    RVO::Vector2 human_pos2(robot_goal);
    RVO::Vector2 human_goal2(robot_pos);

    RVO::Vector2 human_pos3(robot_goal + RVO::Vector2(
        randomize(0.5f, 0.7f), randomize(0.5f, 0.7f)));
    RVO::Vector2 human_goal3(robot_pos + RVO::Vector2(
        randomize(0.5f, 0.7f), randomize(0.5f, 0.7f)));

    RVO::Vector2 human_pos4(robot_pos + RVO::Vector2(
        randomize(0.5f, 0.7f), randomize(0.5f, 0.7f)));
    RVO::Vector2 human_goal4(robot_goal + RVO::Vector2(
        randomize(0.5f, 0.7f), randomize(0.5f, 0.7f)));

    sim->addAgent(human_pos1, 15.0f, 10, 5.0f, 5.0f, randomize(0.12f, 0.22f),
        randomize(0.1f, 0.3f));
    goals.push_back(human_goal1);

    sim->addAgent(human_pos2, 15.0f, 10, 5.0f, 5.0f, randomize(0.12f, 0.22f),
        randomize(0.1f, 0.3f));
    goals.push_back(human_goal2);

    sim->addAgent(human_pos3, 15.0f, 10, 5.0f, 5.0f, randomize(0.12f, 0.22f),
        randomize(0.1f, 0.3f));
    goals.push_back(human_goal3);

    sim->addAgent(human_pos4, 15.0f, 10, 5.0f, 5.0f, randomize(0.12f, 0.22f),
        randomize(0.1f, 0.3f));
    goals.push_back(human_goal4);

    std::vector<std::vector<RVO::Vector2>> empty;
    return empty;
}

int sign(float x) {
    if (x > 0) {
        return 1.0;
    }
    else if (x < 0) {
        return -1.0;
    }
    return 0.0;
}

float truncate(float x, float threshold=0.1) {
    if (abs(x) > 0.1) {
        x = sign(x) * 0.1;
    }
    return x;
}

void updateVisualization(RVO::RVOSimulator *sim, std::ofstream * file)
{
	/* Output the current global time. */
	// std::cout << sim->getGlobalTime();
    *file << sim->getGlobalTime();
	/* Output the current position of all the agents. */

    /* generate gaussian noise to observation */
    float current_x = 0, current_y = 0;
	for (size_t i = 0; i < sim->getNumAgents(); ++i) {
		// std::cout << " " << sim->getAgentPosition(i);
        std::default_random_engine generator;
        std::normal_distribution<float> distribution(0.0, 0.1);
        float noise_x = distribution(generator);
        float noise_y = distribution(generator);
        RVO::Vector2 noise(truncate(noise_x), truncate(noise_y));
        RVO::Vector2 position(sim->getAgentPosition(i));
        RVO::Vector2 velocity = sim->getAgentVelocity(i);
        float radius = sim->getAgentRadius(i);
        *file << " " << position << " " << velocity << " " << radius;
        if (i == 0) {
            current_y = position.y();
            current_x = position.x();
            float v_pref = sim-> getAgentMaxSpeed(i);
            float theta = atan2(velocity.y(), velocity.x());
            *file << " " << goals[i] << " " << v_pref << " "  << theta;
        } 
	}
    
	// std::cout << std::endl;
    *file << std::endl;
}

void setPreferredVelocities(RVO::RVOSimulator *sim)
{
	/*
	 * Set the preferred velocity to be a vector of unit magnitude (speed) in the
	 * direction of the goal.
	 */
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int i = 0; i < static_cast<int>(sim->getNumAgents()); ++i) {
		RVO::Vector2 goalVector = goals[i] - sim->getAgentPosition(i);

		if (RVO::absSq(goalVector) > 1.0f) {
			goalVector = RVO::normalize(goalVector);
		}

		sim->setAgentPrefVelocity(i, goalVector);

		/*
		 * Perturb a little to avoid deadlocks due to perfect symmetry.
		 */
		float angle = std::rand() * 2.0f * M_PI / RAND_MAX;
		float dist = std::rand() * 0.0001f / RAND_MAX;

		sim->setAgentPrefVelocity(i, sim->getAgentPrefVelocity(i) +
		                          dist * RVO::Vector2(std::cos(angle), std::sin(angle)));
	}
}

bool reachedGoal(RVO::RVOSimulator *sim)
{
	/* Check if all agents have reached their goals. */
	if (RVO::absSq(sim->getAgentPosition(0) - goals[0]) <= sim->getAgentRadius(0)) {
        return true;
    }
	return false;
}

int main(int argc, char ** argv)
{   
    int start = START_EXP;
    int max_num = MAX_NUM_EXPS;
    if (argc == 3) {
        start = std::stoi(argv[1]);
        max_num = std::stoi(argv[2]);
    }
    for (int i = start; i < max_num + start; i++) {
		std::cout << "Generating episode " << i << " ...." << std::endl;
        std::ofstream *file = new std::ofstream();
        std::string filename = "data/";
        filename = filename + "Overtake_" + std::to_string(i) + ".txt";
        (*file).open (filename);

        /* Create a new simulator instance. */
        RVO::RVOSimulator *sim = new RVO::RVOSimulator();
        /* Set up the scenario. */
        std::vector<std::vector<RVO::Vector2> > obstacles = setupScenario(sim);
        *file << "[";
        for (size_t j = 0; j<obstacles.size(); j++) {
            *file << "[";
            for (size_t k = 0; k < obstacles[j].size(); k++) {
                (*file) << "(" << obstacles[j][k].x() << ", "
                    << obstacles[j][k].y() << ")";
                if(k < obstacles[j].size()-1){
                    *file << ", ";
                }
            }
            *file << "]";
            if(j < obstacles.size()-1){
                *file << ", ";
            }
        }
        *file << "]" << std::endl;
        *file << "timestamp";
        for (size_t i = 0; i < sim->getNumAgents(); ++i) {
            *file << " position" << i << " velocity" << i << " radius" << i;
            if (i == 0) {
                *file << " goal pref_speed theta";
            }
        }
        int num_agents = sim->getNumAgents();
        for (int i = 0; i < 2; ++i) {
            *file << " position" << num_agents + i << " velocity"
                << num_agents + i << " radius" << num_agents + i;
        }
            
        *file << std::endl;

        /* Perform (and manipulate) the simulation. */
        do {
            updateVisualization(sim, file);
            setPreferredVelocities(sim);
            sim->doStep();
        }
        while (!reachedGoal(sim));
        file->close();
        delete sim;

        std::string obstacle_filename = "data/Overtake_obstacles_" +
            std::to_string(i) + ".txt";
        std::ofstream *obstacle_file = new std::ofstream();
        (*obstacle_file).open(obstacle_filename);

        for (std::vector<RVO::Vector2> obstacle : obstacles) {
            for (RVO::Vector2 vertex : obstacle) {
                (*obstacle_file) << vertex.x() << " " << vertex.y() << " ";
            }
            (*obstacle_file) << std::endl;
        }
        obstacle_file->close();
    }
	return 0;
}
