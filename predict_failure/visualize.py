"""visualize.py
Creates a visualization of the test data output. This module expects to read
a text file that has lines in groups of 3 of the following form:
mux muy sx sy corr ax ay    <-- The prediction made by the predictor network
[(pos0x, pos0y, rad0), (pos1x, pos1y, rad1), ...]   <-- List of initial agent
    positions and radii
[[v11, v12, v13, ...], [v21, v22, v23, ...], ...] <-- List of obstacle
    verticies
This type of file is output by `test.py`. To visualize the network's
predictions, this module draws the initial configuration of the scene and
adds in an ellipse representing the prediction (centered on (mux, muy) with
the size and orientation of the ellipse dictated by the covariance matrix)
as well as the actual final position of the robot.
"""

import matplotlib.pyplot as plt
import argparse
import re
import math
import numpy as np
from matplotlib import patches


def main():
    parser = argparse.ArgumentParser(description="Visualizer for prediction "
                                                 "tester output")
    parser.add_argument("path", type=str, help="path to the file to visualize")
    parser.add_argument("--i", type=int, help="index of visualization to "
                                              "display", default=0)
    parser.add_argument("--noaxis", action="store_true", help="display axis?")
    args = parser.parse_args()
    try:
        file = open(args.path, "r")
        lines = file.readlines()
        print("Total recorded tests: {}".format(int(len(lines)/3)))
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111)
        ax.axis('equal')
        if 0 <= args.i < len(lines)/3:
            # Convert all the parameters of the prediction/actual result
            # back into floats
            pred = [float(x) for x in lines[3*args.i].strip().split("\t")]
            # Separate out all the triples and convert them back into tuples
            # of the form ((pos), rad)
            matches = re.findall(r"(\([0-9\-.+e]+, [0-9\-.+e]+, "
                                r"[0-9\-.+e]+, [0-9\-.+e]+\))",
                                 lines[3*args.i+1].strip())
            agents = []
            for m in matches:
                m = m.lstrip("(").rstrip(")").split(", ")
                agents.append(((float(m[0]), float(m[1])), float(m[2]),
                               float(m[3])))
            matches = re.findall(r"\[[^\[\]]+?\]",
                                 lines[3*args.i+2].strip())
            # Build up the obstacles from sets of points
            obstacles = []
            for m in matches:
                points = re.findall(r"\(([0-9\-.+e]+, [0-9\-.+e]+)\)", m)
                obstacle = []
                for p in points:
                    tmp = p.split(", ")
                    obstacle.append((float(tmp[0]), float(tmp[1])))
                obstacles.append(obstacle)
            # Display all the agents and obstacles, as well as the
            # prediction and actual result.
            agent_colors = [(0.2, 0.72, 0)]
            dot_size = 0.05
            for i in range(len(agents) - 1):
                agent_colors.append((0.0, 0.1, i/(len(agents) - 1)))
            obstacle_color = (0.0, 0.0, 0.0)
            for i in range(len(agents)):
                e = patches.Ellipse((agents[i][0]), 2 * agents[i][1],
                                        2 * agents[i][1],
                                        color=agent_colors[i],
                                    linewidth=2.0, fill=False)
                ax.add_patch(e)
                hx = agents[i][0][0] + agents[i][1] * math.cos(agents[i][2])
                hy = agents[i][0][1] + agents[i][1] * math.sin(agents[i][2])
                h_width = 0.05
                ax.add_patch(patches.Ellipse(
                    (hx, hy), h_width * 2, h_width * 2,
                    color=agent_colors[i], linewidth=2.0, fill=False
                ))
                ax.add_patch(patches.Ellipse(
                    agents[i][0], dot_size, dot_size,
                    color=agent_colors[i], fill=True
                ))
            for i in range(len(obstacles)):
                ax.add_patch(patches.Polygon(
                    obstacles[i], closed=True, color=obstacle_color,
                    fill=False, linewidth=2.0
                ))
            # Show actual endpoint of robot
            actual_color = (0.3, 1.0, 0)
            """ax.add_patch(
                patches.Ellipse((pred[5], pred[6]), 2 * agents[0][1],
                               2 * agents[0][1], color=actual_color,
                               fill=True, linewidth=2.0)
            )"""
            ax.add_patch(patches.Rectangle(
                (agents[0][0][0] + pred[5] - dot_size/2, agents[0][0][1] +
                 pred[6] - dot_size/2), dot_size,
                dot_size, color=(0, 0, 0), fill=True
            ))
            # Draw error ellipse for prediction
            pred_color = (0.2, 0.72, 0)
            cov = pred[4] * pred[3] * pred[2]   # cov = corr * sx * sy
            # Build covariance matrix
            cov_mat = np.array([[pred[2]**2, cov], [cov, pred[3]**2]])
            # Returns tuple containing 2 elements: np array of eigenvalues,
            # then np matrix where cols are eigenvectors
            eig = np.linalg.eig(cov_mat)
            ind = 0
            if math.fabs(eig[0][1]) > math.fabs(eig[0][0]):
                ind = 1
            # Find angle of larger eigenvector in degrees
            ang = math.degrees(math.atan2(eig[1][1][ind], eig[1][0][ind]))
            magic_num = 9.210   # Used to get 99% confidence contour
            # Add the robot's heading back to the dx dy prediction
            r_ang = agents[0][2]
            dx = pred[0] * math.cos(r_ang) - pred[1] * math.sin(r_ang)
            dy = pred[0] * math.sin(r_ang) + pred[1] * math.cos(r_ang)
            ax.add_patch(patches.Ellipse(
                (agents[0][0][0] + dx, agents[0][0][1] + dy),
                2 * math.sqrt(magic_num * eig[0][ind]),
                2 * math.sqrt(magic_num * eig[0][1-ind]), angle=ang,
                fill=False, color=actual_color, linewidth=2.0, linestyle="--"
            ))
            # Put a small point at the mean
            """ax.add_patch(patches.Ellipse(
                (pred[0], pred[1]), dot_size, dot_size, fill=True,
                color=actual_color
            ))"""
            if args.noaxis:
                plt.axis('off')
            plt.xlim((-5, 5))
            plt.ylim((-5, 5))
            plt.show()
        else:
            print("The provided index is out of range (0-{})".format(len(
                lines)/3 - 1))
    except IOError:
        print("Couldn't open file at {}".format(args.path))
        return


if __name__ == "__main__":
    main()
