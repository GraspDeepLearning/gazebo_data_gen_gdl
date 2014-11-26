import h5py
import os
import numpy as np
import math
import matplotlib.pyplot as plt



in_dataset = h5py.File("out.h5")
out_dataset = h5py.File("out_condensed.h5")

labels = in_dataset['rgbd_patch_labels'][:]

label_indices = []

for i in range(labels.shape[0]):
    label_indices.append(np.argmax(labels[i])/4)


hist, indices, patches = plt.hist(label_indices, bins=range(2048/4))

indices_to_keep = indices[hist > 10]
num_patches = sum(hist[indices_to_keep])
num_labels = len(indices_to_keep)*4


out_dataset.create_dataset('rgbd_patches', (num_patches, 72, 72, 4), chunks=(10, 72, 72, 4))
out_dataset.create_dataset('rgbd_patch_labels', (num_patches, num_labels), chunks=(1000, num_labels))

current_index = 0
for i in range(labels.shape[0]):
    label = labels[i]
    label_index = np.where(indices_to_keep == np.argmax(label)/4)
    if len(label_index[0]) > 0:
        label_index = label_index[0][0]
        out_dataset['rgbd_patches'][current_index] = in_dataset['rgbd_patches'][i]
        condensed_label = np.zeros(num_labels)
        condensed_label[label_index*4 + (np.argmax(label) % 4)] = 1
        out_dataset['rgbd_patch_labels'][current_index] = condensed_label

        current_index += 1



import IPython
IPython.embed()