import matplotlib.pyplot as plt
import numpy as np

# record_3_27_20_18.csv : full object, 0.7 dropout, 1024 point

with open("record_3_27_20_18.csv", "r") as f:
    lines = [i.strip("\n").split(", ") for i in f.readlines()]

lines = [[float(num) for num in row] for row in lines]
lines = np.array(lines)

t = list(range(len(lines[:, 0])))
plt.subplot(1, 2, 1)
plt.plot(t, lines[:, 0])
plt.title("Accuracy")
plt.xlabel("epoch")

plt.subplot(1, 2, 2)
plt.plot(t, lines[:, 1])
plt.title("loss")
plt.xlabel("epoch")
plt.show()