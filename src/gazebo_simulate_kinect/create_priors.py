import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import os

#dataset = h5py.File(os.path.expanduser('~/grasp_deep_learning/data/unprocessed_training_data/gdl_7vc.h5'))
dataset = h5py.File('out_condensed.h5')
priors_dataset = h5py.File('priors.h5')

NUM_GRASP_TYPES = dataset['grasp_type_id'][:].max()+1
NUM_VC = dataset['uvd'].shape[1]

#the number of patches we have
NUM_PATCHES = dataset['grasp_type_id'].shape[0]

X_DIM = 480
Y_DIM = 640

priors_dataset.create_dataset('priors',
                              (NUM_GRASP_TYPES, NUM_VC, NUM_VC, X_DIM, Y_DIM),
                              chunks=(1, NUM_VC, NUM_VC, X_DIM, Y_DIM))

priors = priors_dataset['priors']

print "Building Priors"
for index in range(NUM_PATCHES):
    print index

    #the set of uvds for a grasp
    uvds = dataset['uvd'][index]
    #the label for that grasp
    grasp_type_id = dataset['grasp_type_id'][index][0]
    print "grasp_type_id: " + str(grasp_type_id)

    for i in range(NUM_VC):
        for j in range(NUM_VC):

            u0, v0, d0 = uvds[i]
            u1, v1, d1 = uvds[j]

            offset_u = u1-u0
            offset_v = v1-v0

            # import IPython
            # IPython.embed()
            # assert False

            priors[grasp_type_id, i, j, X_DIM/2.0+offset_u, Y_DIM/2.0+offset_v] += 1

print "Blurring Priors"
for index in range(int(NUM_GRASP_TYPES)):
    for i in range(NUM_VC):
        for j in range(NUM_VC):
            # import IPython
            # IPython.embed()
            # assert False

            priors[index, i, j] = scipy.ndimage.gaussian_filter(priors[index, i, j], sigma=3)


##################################################################################
#3added to plot priors
import math
class format_subplot():

    def __init__(self, ax, img):
        self.ax = ax
        self.img = img
        self.ax.format_coord = self.format_coord

    def format_coord(self, x, y):

        return "x=" + str(x) + "  y=" + str(y) + "  z=" + str(self.img[int(y),int(x)])


class Plotter():

    def __init__(self, figure_num=0):
        self.figure_num = figure_num
        self.subplots = []
        self.histograms = []

    def add_subplot(self, title, img):
        self.subplots.append((title, img))

    def add_histogram(self, title, img):
        self.histograms.append((title, img))

    def show(self):
        figure = plt.figure(self.figure_num)
        num_histograms = len(self.histograms)
        num_subplots = len(self.subplots)
        y_dim = 4.0
        x_dim = math.ceil((num_subplots + num_histograms)/y_dim)

        for i in range(len(self.subplots)):
            title, img = self.subplots[i]

            print "plotting: " + str(title)
            print img.shape

            ax = plt.subplot(x_dim, y_dim, i + 1)
            format_subplot(ax, img)
            plt.title(title)
            plt.imshow(img)

        for i in range(len(self.histograms)):
            title, img = self.histograms[i]

            print "plotting: " + str(title)
            print img.shape

            plt.subplot(x_dim,y_dim, num_subplots + i + 1)
            plt.title(title)
            plt.hist(img, bins=10, alpha=0.5)
        plt.show()


plotter = Plotter()

for i in range(int(NUM_GRASP_TYPES)):
    for j in range(NUM_VC):
        for k in range(NUM_VC):
            plotter.add_subplot(str(i) + "_" +str(j) + "_" +str(k), priors[i,j, k])

plotter.show()

import IPython
IPython.embed()