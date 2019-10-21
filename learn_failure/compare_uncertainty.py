"""compare_uncertainty.py
Take in a directory of uncertainty files that contain data and model
uncertainties of a success model in the order dx dy dcorr mx my mcorr and
output a file containing the rolling averages of each file's uncertainty in
different columns.
"""

import argparse
from glob import glob


def main():
    parser = argparse.ArgumentParser(description="Process uncertainties "
                                                 "of a success controller")
    parser.add_argument("directory_success", type=str,
                        help="directory to read familiar data from")
    parser.add_argument("directory_failure", type=str,
                        help="directory to read unfamiliar data from")
    parser.add_argument("-w", type=int, help="size of rolling avg window",
                        default=10)
    parser.add_argument("-o", type=str, help="output file name",
                        default="compout.txt")
    args = parser.parse_args()
    s_path = args.directory_success + "/std_devs_*.txt"
    f_path = args.directory_failure + "/std_devs_*.txt"
    files = glob(s_path)
    f_files = glob(f_path)
    files.extend(f_files)
    # Lists of lists of rolling average values
    mxs = []
    mys = []
    dxs = []
    dys = []
    txs = []
    tys = []
    squares = []
    max_squares = []
    for name in files:
        mxbuff = [0] * args.w
        mybuff = [0] * args.w
        dxbuff = [0] * args.w
        dybuff = [0] * args.w
        # For this file, calculate the rolling averages
        mx_sum = 0
        my_sum = 0
        dx_sum = 0
        dy_sum = 0
        mx = []
        my = []
        dx = []
        dy = []
        tx = []
        ty = []
        square = []
        with open(name, "r") as file:
            lines = file.readlines()
            for i, line in enumerate(lines):
                line = line.split(" ")
                buff_ind = i%args.w
                mx_sum -= mxbuff[buff_ind]
                my_sum -= mybuff[buff_ind]
                dx_sum -= dxbuff[buff_ind]
                dy_sum -= dybuff[buff_ind]
                mxbuff[buff_ind] = float(line[0])
                mybuff[buff_ind] = float(line[1])
                dxbuff[buff_ind] = float(line[3])
                dybuff[buff_ind] = float(line[4])
                last_ind = buff_ind
                mx_sum += mxbuff[buff_ind]
                my_sum += mybuff[buff_ind]
                dx_sum += dxbuff[buff_ind]
                dy_sum += dybuff[buff_ind]
                mx.append(mx_sum / args.w)
                my.append(my_sum / args.w)
                dx.append(dx_sum / args.w)
                dy.append(mx_sum / args.w)
                tx_sum = mx_sum + dx_sum
                ty_sum = my_sum + dy_sum
                tx.append(tx_sum / args.w)
                ty.append(ty_sum / args.w)
                square.append(tx_sum**2 + ty_sum**2)
            mxs.append(mx)
            mys.append(my)
            dxs.append(dx)
            dys.append(dy)
            txs.append(tx)
            tys.append(ty)
            squares.append(square)
            max_squares.append([max(square)])
    vals = [mxs, mys, dxs, dys, txs, tys, squares, max_squares]
    names = ["mx", "my", "dx", "dy", "tx", "ty", "square", "max_squares"]
    for i, val in enumerate(vals):
        with open(names[i] + "_" + args.o, "w") as outfile:
            for j in range(len(val[0])):
                for k in range(len(val)):
                    outfile.write("{} ".format(val[k][j]))
                outfile.write("\n")


if __name__ == "__main__":
    main()
