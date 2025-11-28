import numpy as np


d1 = np.load("original.npy")
d2 = np.load("test.npy")


print(np.max((d1-d2)**2))

from matplotlib import pyplot as plt


for i in range(3):
    plt.subplot(3, 1, i+1)
    plt.plot(d1.T[i])
    plt.plot(d2.T[i])
plt.show()