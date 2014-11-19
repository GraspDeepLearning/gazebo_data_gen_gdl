import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import os

dataset = h5py.File(os.path.expanduser('~/grasp_deep_learning/data/unprocessed_training_data/gdl_barrett.h5'))
priors_dataset = h5py.File('priors.h5')

NUM_GRASP_TYPES = 4*4*4*4
NUM_VC = dataset['uvd'].shape[1]

X_DIM = 480
Y_DIM = 640

priors_dataset.create_dataset('priors',
                              (NUM_GRASP_TYPES, NUM_VC, NUM_VC, X_DIM, Y_DIM),
                              chunks=(1, NUM_VC, NUM_VC, X_DIM, Y_DIM))

priors = priors_dataset['priors']

print "Building Priors"
for index in range(NUM_GRASP_TYPES):
    print index

    uvds = dataset['uvd'][index]
    for i in range(NUM_VC):
        for j in range(NUM_VC):

            u0, v0, d0 = uvds[i]
            u1, v1, d1 = uvds[j]

            offset_u = u1-u0
            offset_v = v1-v0

            priors[index, i, j, X_DIM/2.0+offset_u, Y_DIM/2.0+offset_v] += 1

print "Blurring Priors"
for index in range(NUM_GRASP_TYPES):
    for i in range(NUM_VC):
        for j in range(NUM_VC):
            priors[index, i, j] = scipy.ndimage.gaussian_filter(priors[index, i, j], sigma=3)


import IPython
IPython.embed()