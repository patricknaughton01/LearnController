/*
 * Blocks.cpp
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
 * Example file showing a demo with 100 agents split in four groups initially
 * positioned in four corners of the environment. Each agent attempts to move to
 * other side of the environment through a narrow passage generated by four
 * obstacles. There is no roadmap to guide the agents around the obstacles.
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

#define START_EXP 0
#define MAX_NUM_EXPS 100

#define WALL_WIDTH 1.0f
#define WALL_DIST 4.0f
#define WALL_LENGTH 7.0f
#define VERTICAL_WALL false
#define NUM_PEOPLE 4.0f

/* Store the goals of the agents. */
std::vector<RVO::Vector2> goals;
// Store the headings of the agents
std::vector<double> headings;

float randomize(float LO=0.0f, float HI=1.0f)
{
    return LO + static_cast <float> (rand()) / static_cast <float> (RAND_MAX / (HI - LO));
}

std::vector<std::vector<RVO::Vector2> > setupScenario(RVO::RVOSimulator *sim)
{
	// std::srand(static_cast<unsigned int>(std::time(NULL)));

	/* Specify the global time step of the simulation. */
	sim->setTimeStep(1.0f);
	//sim->setTimeStep(0.25f);
	//sim->setTimeStep(0.f);
    double heading_range = M_PI/8.0;

	/*
	 * Add agents, specifying their start position, and store their goals on the
	 * opposite side of the environment.
	 */

    std::vector<std::vector<RVO::Vector2> > obstacles;
    float wall_length = WALL_LENGTH, wall_width = WALL_WIDTH, wall_dist = WALL_DIST;
    if (VERTICAL_WALL) {
        std::vector<RVO::Vector2> left_wall_vertices, right_wall_vertices;
        left_wall_vertices.push_back(RVO::Vector2(wall_width, 0));
        left_wall_vertices.push_back(RVO::Vector2(wall_width, wall_length));  
        left_wall_vertices.push_back(RVO::Vector2(0, wall_length));  
        left_wall_vertices.push_back(RVO::Vector2(0, 0));

        right_wall_vertices.push_back(RVO::Vector2(2 * wall_width + wall_dist, 0));
        right_wall_vertices.push_back(RVO::Vector2(2 * wall_width + wall_dist, wall_length)); 
        right_wall_vertices.push_back(RVO::Vector2(wall_width + wall_dist, wall_length));   
        right_wall_vertices.push_back(RVO::Vector2(wall_width + wall_dist, 0));
        
        sim->addObstacle(left_wall_vertices);
        sim->addObstacle(right_wall_vertices);
        sim->processObstacles();

        obstacles.push_back(left_wall_vertices);
        obstacles.push_back(right_wall_vertices);

        RVO::Vector2 robot_pos(randomize(-0.2f + wall_width + wall_dist / 2.0, wall_width + wall_dist / 2.0 + 0.2f), 
                            randomize(0.8f, 1.4f));
        RVO::Vector2 robot_goal(robot_pos.x(), robot_pos.y() + randomize(4.0f, 5.0f));

        sim->addAgent(robot_pos, 1.0f, 10, 5.0f, 5.0f, randomize(0.15f, 0.22f), randomize(1.5f, 2.0f));
        goals.push_back(robot_goal);
        headings.push_back(randomize(-heading_range, heading_range));
        
        RVO::Vector2 human_pos1(randomize(wall_width + 0.1f, wall_width + wall_dist / NUM_PEOPLE - 0.1f), 
                                randomize(wall_length + 0.2f, wall_length + 0.7f));
        RVO::Vector2 human_goal1(randomize(-0.5f, wall_width), 
                                randomize(wall_length + 1.2f, wall_length + 2.2f));

        RVO::Vector2 human_pos2(randomize(wall_width + wall_dist / NUM_PEOPLE + 0.1f, wall_width + wall_dist / NUM_PEOPLE * 2 - 0.1f), 
                                randomize(wall_length - 0.2f, wall_length + 0.3f));
        RVO::Vector2 human_goal2(randomize(wall_width, wall_width + wall_dist / NUM_PEOPLE), 
                                randomize(wall_length + 1.2f, wall_length + 2.2f));   

        RVO::Vector2 human_pos3(randomize(wall_width + wall_dist / NUM_PEOPLE * 2 + 0.1f, wall_width + wall_dist - 0.1f), 
                                randomize(wall_length + 0.2f, wall_length + 0.7f));
        RVO::Vector2 human_goal3(randomize(wall_width * 2 + wall_dist, wall_width * 2 + wall_dist + 1.0f), 
                                randomize(wall_length + 1.2f, wall_length + 2.2f));   
        
        RVO::Vector2 human_pos4(randomize(wall_width + wall_dist / NUM_PEOPLE * 3 + 0.1f, wall_width + wall_dist - 0.1f), 
                                randomize(wall_length - 0.2f, wall_length + 0.3f));
        RVO::Vector2 human_goal4(randomize(wall_width * 2 + wall_dist, wall_width * 2 + wall_dist + 1.0f), 
                                randomize(wall_length + 1.2f, wall_length + 2.2f));   
        
        sim->addAgent(human_pos1, 1.0f, 10, 5.0f, 5.0f, randomize(0.12f, 0.22f), randomize(0.1f, 0.3f));
        goals.push_back(human_goal1);
        headings.push_back(randomize(-heading_range, heading_range));

        sim->addAgent(human_pos2, 1.0f, 10, 5.0f, 5.0f, randomize(0.12f, 0.22f), randomize(0.1f, 0.3f));
        goals.push_back(human_goal2);
        headings.push_back(randomize(-heading_range, heading_range));

        sim->addAgent(human_pos3, 1.0f, 10, 5.0f, 5.0f, randomize(0.12f, 0.22f), randomize(0.1f, 0.3f));
        goals.push_back(human_goal3);
        headings.push_back(randomize(-heading_range, heading_range));

        sim->addAgent(human_pos4, 1.0f, 10, 5.0f, 5.0f, randomize(0.12f, 0.22f), randomize(0.1f, 0.3f));
        goals.push_back(human_goal4);
        headings.push_back(randomize(-heading_range, heading_range));
    }
    else {
        std::vector<RVO::Vector2> up_wall_vertices, down_wall_vertices;
        up_wall_vertices.push_back(RVO::Vector2(wall_length, 2 * wall_width +
        wall_dist));
        up_wall_vertices.push_back(RVO::Vector2(0, 2 * wall_width + wall_dist));
        up_wall_vertices.push_back(RVO::Vector2(0, wall_dist + wall_width));
        up_wall_vertices.push_back(RVO::Vector2(wall_length, wall_dist +
        wall_width));

        down_wall_vertices.push_back(RVO::Vector2(wall_length, wall_width));
        down_wall_vertices.push_back(RVO::Vector2(0, wall_width)); 
        down_wall_vertices.push_back(RVO::Vector2(0, 0));   
        down_wall_vertices.push_back(RVO::Vector2(wall_length, 0));
        
        sim->addObstacle(up_wall_vertices);
        sim->addObstacle(down_wall_vertices);
        sim->processObstacles();

        obstacles.push_back(up_wall_vertices);
        obstacles.push_back(down_wall_vertices);

//        RVO::Vector2 robot_pos(randomize(1.0f, 1.5f),
//                               randomize(-0.15f + wall_width + wall_dist / 2.0, wall_width + wall_dist / 2.0 + 0.15f));
        RVO::Vector2 robot_pos(wall_length-1.0f, wall_width + wall_dist/2.0
            + randomize(-0.5, 0.5));
        RVO::Vector2 robot_goal(wall_length + 3.0, wall_width + wall_dist/2.0
            + randomize(-0.5, 0.5));

        RVO::Vector2 human_pos1(randomize(wall_length, wall_length + 0.5f),
                                wall_width + (wall_dist/NUM_PEOPLE)/2.0f);
        RVO::Vector2 human_goal1(human_pos1.x() + randomize(wall_length+3-0.2f, wall_length+3+0.2f),
                                 human_pos1.y() - 1.0f);

        RVO::Vector2 human_pos2(randomize(wall_length+1.0f, wall_length + 1.5f),
                                human_pos1.y()+1.0f);

        RVO::Vector2 human_goal2(human_pos2.x() + randomize(wall_length+3-0.2f, wall_length+3+0.2f),
                                 human_pos2.y() - 0.5f);

        RVO::Vector2 human_pos3(randomize(wall_length, wall_length + 0.5f),
                                human_pos2.y()+1.0f);

        RVO::Vector2 human_goal3(human_pos3.x() + randomize(wall_length+3-0.2f, wall_length+3+0.2f),
                                 human_pos3.y() + 0.5f);

        RVO::Vector2 human_pos4(randomize(wall_length+1.0f, wall_length + 1.5f),
                                 human_pos3.y()+1.0f);

        RVO::Vector2 human_goal4(human_pos4.x() + randomize(wall_length+3-0.2f, wall_length+3+0.2f),
                                human_pos4.y() + 1.0f);
        

        sim->addAgent(robot_pos, 10.0f, 10, 1.0f, 5.0f, 0.5f, 3.0f);
        goals.push_back(robot_goal);
        headings.push_back(randomize(-heading_range, heading_range));

        sim->addAgent(human_pos1, 10.0f, 10, 1.0f, 5.0f, 0.5f, 0.7f);
        goals.push_back(human_goal1);
        headings.push_back(randomize(-heading_range, heading_range));

        sim->addAgent(human_pos2, 10.0f, 10, 1.0f, 5.0f, 0.5f, 0.7f);
        goals.push_back(human_goal2);
        headings.push_back(randomize(-heading_range, heading_range));

        sim->addAgent(human_pos3, 10.0f, 10, 1.0f, 5.0f, 0.5f, 0.7f);
        goals.push_back(human_goal3);
        headings.push_back(randomize(-heading_range, heading_range));

        sim->addAgent(human_pos4, 10.0f, 10, 1.0f, 5.0f, 0.5f, 0.7f);
        goals.push_back(human_goal4);
        headings.push_back(randomize(-heading_range, heading_range));
    }
    

    return obstacles;
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
        *file << " " << position << " " << velocity << " " << radius <<
            " " << headings[i];
        if (i == 0) {
            current_y = position.y();
            current_x = position.x();
            float v_pref = sim-> getAgentMaxSpeed(i);
            float theta = atan2(velocity.y(), velocity.x());
            *file << " " << goals[i] << " " << v_pref << " "  << theta;
        } 
	}
    if (VERTICAL_WALL) {
        /*
        The wall is like this:
        |  |
        |  |
        |  |
        |  |
        */
        RVO::Vector2 left_closest_point(0, std::min(std::max(current_y, 0.0f), WALL_LENGTH));
        RVO::Vector2 right_closest_point(2 * WALL_WIDTH + WALL_DIST, std::min(std::max(current_y, 0.0f), WALL_LENGTH));
        RVO::Vector2 static_vel(0, 0);
        /*
        *file << " " << left_closest_point << " " << static_vel << " " << WALL_WIDTH;
        *file << " " << right_closest_point << " " << static_vel << " " << WALL_WIDTH;
        */
        *file << " " << left_closest_point << " " << static_vel << " "
            << 0.001f << " " << headings[0];
        *file << " " << right_closest_point << " " << static_vel << " "
            << 0.001f << " " << headings[0];
    }
    else {
        /*
        The wall is like this:
        -------------
        
        -------------
        */
        RVO::Vector2 up_closest_point(std::min(std::max(current_x, 0.0f),
            WALL_LENGTH), WALL_WIDTH + WALL_DIST);
        RVO::Vector2 down_closest_point(std::min(std::max(current_x, 0.0f),
            WALL_LENGTH), WALL_WIDTH);
        RVO::Vector2 static_vel(0, 0);
        /*
        *file << " " << up_closest_point << " " << static_vel << " " << WALL_WIDTH;
        *file << " " << down_closest_point << " " << static_vel << " " << WALL_WIDTH;
        */
        *file << " " << up_closest_point << " " << static_vel << " "
            << 0.001f << " " << headings[0];
        *file << " " << down_closest_point << " " << static_vel << " "
            << 0.001f << " " << headings[0];
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
        filename = filename + "Barge_in_" + std::to_string(i) + ".txt";
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
            *file << " position" << i << " velocity" << i << " radius" << i <<
                " heading" << i;
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
        updateVisualization(sim, file);
        // Do 10 more steps
//        for(int i = 0; i<10; i++){
//            updateVisualization(sim, file);
//            setPreferredVelocities(sim);
//            sim->doStep();
//        }
        file->close();
        delete sim;

        std::string obstacle_filename = "data/Barge_in_obstacles_" + std::to_string(i) + ".txt";
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
