import matplotlib.pyplot as plt
import numpy as np


# record_3_27_22_2.csv : full object, 0.7 dropout, 4 point..??
# record_3_28_13_33.csv : full object, 0.7 dropout, 8 point
# record_3_28_13_30.csv : full object, 0.7 dropout, 16 point
# record_3_27_22_1.csv : full object, 0.7 dropout, 32 point
# record_3_27_22_0.csv : full object, 0.7 dropout, 64 point
# record_3_27_21_57.csv : full object, 0.7 dropout, 128 point
# record_3_27_21_55.csv : full object, 0.7 dropout, 256 point
# record_3_27_21_52.csv : full object, 0.7 dropout, 512 point
# record_3_27_20_18.csv : full object, 0.7 dropout, 1024 point
# record_3_27_20_38.csv : full object, 0.7 dropout, 2048 point

points = 1024
dropout = 0.7

with open("record_3_27_20_18.csv", "r") as f:
    lines = [i.strip("\n").split(", ") for i in f.readlines()]

lines = [[float(num) for num in row] for row in lines]
lines = np.array(lines)

t = list(range(len(lines[:, 0])))
plt.suptitle(f"num points: {points}, dropout: {dropout}")
plt.subplot(1, 2, 1)
plt.plot(t, lines[:, 0])
plt.title("Accuracy")
plt.xlabel("epoch")

plt.subplot(1, 2, 2)
plt.plot(t, lines[:, 1], "r")
plt.title("loss")
plt.xlabel("epoch")
plt.show()
