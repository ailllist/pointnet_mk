import matplotlib.pyplot as plt
import numpy as np

dict_ = {1024: [82.48885285772192, 0.6877557635307312], 2048: [82.2051074179165, 0.6641191244125366],
         512: [82.28617754357519, 0.6657659411430359], 256: [81.71868666396432, 0.652273416519165],
         128: [79.48925820835022, 0.7231566905975342], 64: [74.17916497770571, 0.856777012348175],
         32: [72.63883259019052, 0.9372515678405762], 4: [38.58937981353871, 2.1413254737854004],
         16: [71.46331576813945, 0.9999222755432129], 8: [61.16740980948521, 1.4141080379486084],
         2: [25.699229833806243, 2.7384278774261475]}

keys = dict_.keys()
n_dict = {}

for i in sorted(keys, reverse=True):
    n_dict["%.3f" % float((2048 - i)/2048)] = dict_[i]

missing_p = n_dict.keys()
values = n_dict.values()

missing = [float(i) for i in missing_p]
a_l = np.array(list(values))

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
fig.subplots_adjust(hspace=0.1)

ax1.plot(missing, a_l[:, 0], "-^")
ax2.plot(missing, a_l[:, 1], "r-*")
ax1.set_ylim(20, 90)
ax2.set_ylim(0, 3)

ax1.spines["bottom"].set_visible(False)
ax2.spines["top"].set_visible(False)

ax1.xaxis.tick_top()
ax1.tick_params(labeltop=True)

kwargs = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

ax1.set_ylabel("Accuracy (%)")
ax2.set_ylabel("loss")
ax2.set_xlabel("missing rate")
plt.suptitle("incomplete input data")
ax1.legend(["accuracy"])
ax2.legend(["loss"])

plt.show()
