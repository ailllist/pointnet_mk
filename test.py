import numpy as np
import glob
import os

point_dict = {"record_3_28_14_5.csv": 2,
              "record_3_27_22_2.csv": 4,
              "record_3_28_13_33.csv": 8,
              "record_3_28_13_30.csv": 16,
              "record_3_27_22_1.csv": 32,
              "record_3_27_22_0.csv": 64,
              "record_3_27_21_57.csv": 128,
              "record_3_27_21_55.csv": 256,
              "record_3_27_21_52.csv": 512,
              "record_3_27_20_18.csv": 1024,
              "record_3_27_20_38.csv": 2048
              }


def max_min(file_name):
    with open(file_name, "r") as f:
        lines = [i.strip("\n").split(", ") for i in f.readlines()]

    lines = [[float(num) for num in row] for row in lines]
    lines = np.array(lines)

    acc = max(lines[:, 0][10:])
    loss = min(lines[:, 1][10:])

    return acc, loss


DIR = os.path.dirname(os.path.abspath(__file__))
final_dir = {}

for csv_name in glob.glob(os.path.join(DIR, "record_*.csv")):
    acc, loss = max_min(csv_name)
    print(point_dict[csv_name.replace(DIR+"\\", "")])
    print(acc, loss, "\n")

    final_dir[point_dict[csv_name.replace(DIR+"\\", "")]] = [acc, loss]

print(final_dir)