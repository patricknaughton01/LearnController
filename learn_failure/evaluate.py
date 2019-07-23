"""evaluate.py
Script to evaluate recorded trajectories. Expects files to contain a one
line header followed by any number of space separated lines describing the
trajectories of agents. See `tests/barge/barge_fail/*.txt` for examples of
this format. The first set of parameters is assumed to describe the robot.

Note that the `simulator.Simulator.update_visualization` function will adhere
to the expected format.
"""

import argparse
import glob
import math
from utils import dist


def main():
    parser = argparse.ArgumentParser(description="Evaluate trajectories")
    parser.add_argument("path", type=str, help="Path to files containing "
                                               "trajectories", default=".")
    parser.add_argument("--out_path", type=str,
                        help="Path to save evaluations to", default="out.txt")
    parser.add_argument("-v", action="store_true", help="Provide verbose "
                                                        "output")
    args = parser.parse_args()
    file_paths = glob.glob(args.path + "/*.txt")
    distances_list = []
    ang_distances_list = []
    times_list = []
    collisions_list = []
    intrusions_list = []
    out = open(args.out_path, "a")
    out.write("dist\tang_dist\ttime\tcoll\tintrusion\n")
    for path in file_paths:
        try:
            file = open(path, "r")
            lines = file.readlines()
            if len(lines) > 2:
                # Metrics for evaluating the trajectories
                total_dist = 0.0
                total_ang_dist = 0.0
                time_spent = 0.0
                collisions = 0
                intrusions = 0
                intrusion_thresh = 0.2
                for i in range(1, len(lines)-1):
                    last_line = lines[i].strip().split(" ")
                    line = lines[i+1].strip().split(" ")
                    # Check and make sure we have all the parameters for the
                    # robot
                    if len(last_line) >= 7 and len(line) >= 7:
                        last_pos = split_coordinates(last_line[1])
                        r_pos = split_coordinates(line[1])
                        total_dist += dist(last_pos, r_pos)
                        last_theta = float(last_line[6])
                        r_theta = float(line[6])
                        ang_diff = math.fabs(r_theta - last_theta)
                        # Normalize the angle change, i.e. pi ==> -pi should
                        # register as no angle change, not 2pi
                        while ang_diff > math.pi:
                            ang_diff -= math.pi
                        total_ang_dist += ang_diff
                        r_rad = float(line[3])
                        # Examine all agents and obstacles to check for
                        # collisions and/or intrusions
                        for j in range(7, len(line), 3):
                            try:
                                obs_pos = split_coordinates(line[j])
                                obs_rad = float(line[j+2])
                                d = dist(r_pos, obs_pos)
                                if d < r_rad + obs_rad:
                                    collisions += 1
                                elif d < r_rad + obs_rad + intrusion_thresh:
                                    intrusions += 1
                            except IndexError:
                                print("Malformed object, skipping")
                # Look for the last valid line in the file, consider this
                # the final stopping point of the robot. Find how long it
                # has been there (within its radius/10.0) to see how long it
                # took to reach its final position.
                final_pos = None
                for i in range(len(lines)-1, -1, -1):
                    line = lines[i].split(" ")
                    if len(line) > 7 and final_pos is None:
                        final_pos = split_coordinates(line[1])
                    elif final_pos is not None:
                        d = dist(split_coordinates(line[1]), final_pos)
                        if d > float(line[3])/10.0:
                            # Time always starts at 0 so we just need this
                            # timestamp
                            time_spent = float(line[0])
                            break
                if args.v:
                    print("\t".join([str(total_dist), str(total_ang_dist),
                                     str(time_spent), str(collisions),
                                     str(intrusions)]))
                distances_list.append(total_dist)
                ang_distances_list.append(total_ang_dist)
                times_list.append(time_spent)
                collisions_list.append(collisions)
                intrusions_list.append(intrusions)
                out.write("\t".join([str(total_dist), str(total_ang_dist),
                                     str(time_spent), str(collisions),
                                     str(intrusions)]))
                out.write("\n")
            file.close()
        except IOError:
            print("Couldn't open path {}. Skipping...".format(path))
    # Newline then the averages and standard deviations
    out.write("\n")
    avgs = [
        str(avg(distances_list)),
        str(avg(ang_distances_list)),
        str(avg(times_list)),
        str(avg(collisions_list)),
        str(avg(intrusions_list))
    ]
    std_devs = [
        str(std_dev(distances_list)),
        str(std_dev(ang_distances_list)),
        str(std_dev(times_list)),
        str(std_dev(collisions_list)),
        str(std_dev(intrusions_list))
    ]
    out.write("\t".join(avgs))
    out.write("\n")
    out.write("\t".join(std_devs))
    out.write("\n")
    if args.v:
        print("Averages:")
        print("\t".join(avgs))
        print("Standard Deviations:")
        print("\t".join(std_devs))


def avg(vals):
    """Computes the average of the numerical values in `vals`.

    :param iterable vals: iterable of numerical values to average.
    :return: The numerical average of the values in `vals`
        :rtype: float
    """
    if len(vals) > 0:
        total = 0
        for v in vals:
            total += v
        return total / len(vals)
    return None


def std_dev(vals):
    """Computes the (sample) standard deviation of the numerical values in
    `vals`.

    :param iterable vals: Numerical values to find the standard deviation of
    :return: The (sample) standard deviation of the values in `vals`
        :rtype: float
    """
    if len(vals) > 0:
        mean = avg(vals)
        total = 0
        for v in vals:
            total += (v - mean) ** 2
        return math.sqrt(total / (len(vals) - 1))
    return None


def split_coordinates(str_coors):
    """Splits the string `str_coors` into a tuple containing two coordinates,
    x and y.

    :param str str_coors: A string representing coordinates of the form '(x,y)'
    :return: A tuple of x and y coordinates
        :rtype: (float, float)
    """
    tmp = str_coors[1:-1].split(",")
    if len(tmp) > 1:
        return float(tmp[0]), float(tmp[1])
    return None

if __name__ == "__main__":
    main()
