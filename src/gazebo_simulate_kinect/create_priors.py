import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage

dataset = h5py.File("out.h5")

NUM_GRASPS = dataset['labels'].shape[0]
NUM_HEATMAPS = dataset['labels'].shape[1]
X_DIM = 480
Y_DIM = 640

priors = np.zeros((NUM_HEATMAPS, NUM_HEATMAPS, X_DIM, Y_DIM))

for index in range(NUM_GRASPS):
    print index
    for i in range(NUM_HEATMAPS):
        for j in range(NUM_HEATMAPS):
            vc_0 = dataset['labels'][index][i]
            vc_1 = dataset['labels'][index][j]

            arg_max0 = np.argmin(vc_0)
            arg_max1 = np.argmin(vc_1)

            x0, y0 = arg_max0 / vc_0.shape[1], arg_max0 % vc_0.shape[1]
            x1, y1 = arg_max1 / vc_1.shape[1], arg_max1 % vc_1.shape[1]

            offset_x = x1-x0
            offset_y = y1-y0

            priors[i, j, X_DIM/2.0+offset_x, Y_DIM/2.0+offset_y] += 1


for i in range(NUM_HEATMAPS):
    for j in range(NUM_HEATMAPS):
        priors[i, j] = scipy.ndimage.gaussian_filter(priors[i,j], sigma=3)


import IPython
IPython.embed()