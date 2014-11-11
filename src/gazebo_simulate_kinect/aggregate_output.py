import h5py
import os

GDL_DATA_PATH = os.environ["GDL_PATH"] + "data"

input_directory = GDL_DATA_PATH + '/rgbd_images_11_11_13_44'

subdirs = os.listdir(input_directory)


num_patches = 0
num_images = 0
num_heatmaps_per_patch = 0

image_shape = ()
patch_shape = ()

for subdir in subdirs:
    in_dataset_fullpath = input_directory + '/' + subdir + "/rgbd_and_labels.h5"
    print in_dataset_fullpath
    in_dataset = h5py.File(in_dataset_fullpath)

    num_patches += in_dataset['rgbd_patches'].shape[0]
    num_images += in_dataset['rgbd'].shape[0]
    num_heatmaps_per_patch = in_dataset['rgbd_patches'].shape[1]

    #skip image number
    image_shape = in_dataset['rgbd'].shape[1:]

    #skip first indice is image number
    #second indice is heatmap number
    patch_shape = in_dataset['rgbd_patches'].shape[2:]


patches_dataset_size = [num_patches*num_heatmaps_per_patch] + list(patch_shape)
images_dataset_size = [num_images] + list(image_shape)
patch_labels_dataset_size = [num_patches*num_heatmaps_per_patch, num_heatmaps_per_patch]

patches_chunk_size = tuple([10] + list(patch_shape))
images_chunk_size = tuple([10] + list(image_shape))
patch_labels_chunk_size = tuple([10, num_heatmaps_per_patch])

out_dataset = h5py.File("out.h5")
out_dataset.create_dataset("rgbd_patches",  patches_dataset_size, chunks=patches_chunk_size)
out_dataset.create_dataset("rgbd",  images_dataset_size, chunks=images_chunk_size)
out_dataset.create_dataset("rgbd_patch_labels",  patch_labels_dataset_size, chunks=patch_labels_chunk_size)


image_count = 0
patch_count = 0

for subdir in subdirs:

    in_dataset_fullpath = input_directory + '/' + subdir + "/rgbd_and_labels.h5"
    print in_dataset_fullpath
    in_dataset = h5py.File(in_dataset_fullpath)

    for i in range(in_dataset['rgbd_patches'].shape[0]):
        for j in range(num_heatmaps_per_patch):
            out_dataset['rgbd_patches'][patch_count] = in_dataset['rgbd_patches'][i, j]
            out_dataset['rgbd_patch_labels'][patch_count, j] = in_dataset['rgbd_patch_labels'][i, 0]
            patch_count += 1

    for i in range(in_dataset['rgbd'].shape[0]):
        out_dataset['rgbd'][image_count] = in_dataset['rgbd'][i]
        image_count += 1

import IPython
IPython.embed()