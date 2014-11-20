import h5py
import os
import numpy as np
import math
import matplotlib.pyplot as plt

#the number of virtual contacts collected
NUM_VC_IN = 17

#the number of virtual contacts we actually want to keep.
#most likely just palm and fingertips
NUM_VC_OUT = 4

#this are the indices for the virtual contacts we want to keep
PALM_INDEX = 0
FINGER_1_INDEX = 8
FINGER_2_INDEX = 12
FINGER_3_INDEX = 16
VC_INDICES = [PALM_INDEX, FINGER_1_INDEX, FINGER_2_INDEX, FINGER_3_INDEX]

#number of degrees of freedom for the hand
NUM_DOF = 4

#this is the number of bins for each dof of the barret hand.  These are used to classify the grasp type
NUM_BINS = 4

#two bins, for touching or not touching palm to object
NUM_DEPTH_BINS = 2

#the root data directory
GDL_DATA_PATH = os.environ["GDL_PATH"] + "/data"

#the directory we are going to pull all the h5 files from.
INPUT_DIRECTORY = GDL_DATA_PATH + '/rgbd_images/11_17_18_38'


#we have NUM_VC * NUM_BINS**NUM_DOF different labels
#
#dof values = the dof values for this particular grasp:
# ex: [3.01, 1.01, 2.47, 0.45]
#
# bin_edges = for each dof, the edges separating the different bins for that dof:
# ex: [[0, 1, 2, 3],
#      [0, 1, 2, 3],
#      [0, 1, 2, 3],
#      [0, 1, 2, 3]]
#
#vc_id: which virtual contact is this label for?
# ex: 0
#
#index will be
#bin_id0*num_dof**0 + bin_id1*num_dof**1 + bin_id2*num_dof**2 + bin_id3*num_dof**3 + vc_id*num_vc**4
#3*1 + 3*4**1 + 3*4**2 + 3*4**3 + 3* 4**4
def get_label_index(dof_values, bin_edges_list, vc_id, d):

    #helper function to determine what bin a data point belongs in.
    def get_bin(data_point, bin_edges):
        bin_id = 0
        for bin_edge in bin_edges:

            #this will never pass on first bin_edge
            if data_point < bin_edge:
                break

            bin_id += 1

        bin_id -= 1

        #sanity check
        assert bin_id >= 0
        assert bin_id < NUM_BINS

        return bin_id

    label_index = 0
    for i in range(NUM_DOF):

        dof_value = dof_values[i]
        bin_edges = bin_edges_list[i]

        #this should be between 0 and 3 inclusive
        bin_id = get_bin(dof_value, bin_edges)

        label_index += bin_id * math.pow(NUM_BINS, NUM_BINS-i)

    label_index += (vc_id)

    d_bin = 0
    if d > .02:
        d_bin = 1

    label_index += d_bin * 1024

    return label_index


#quickly run throught the directories and determine the shape of the patches
#and how many patches and images we have
def get_data_dimensions(subdirs):
    num_patches = 0
    num_images = 0
    num_heatmaps_per_patch = 0

    image_shape = ()
    patch_shape = ()

    first_subdir = True

    for subdir in subdirs:

        in_dataset_fullpath = INPUT_DIRECTORY + '/' + subdir + "/rgbd_and_labels.h5"
        in_dataset = h5py.File(in_dataset_fullpath)

        num_patches += in_dataset['rgbd_patches'].shape[0]
        num_images += in_dataset['rgbd'].shape[0]

        if first_subdir:
            num_heatmaps_per_patch = in_dataset['rgbd_patches'].shape[1]

            #skip image number
            image_shape = in_dataset['rgbd'].shape[1:]

            #skip first indice is image number
            #second indice is heatmap number
            patch_shape = in_dataset['rgbd_patches'].shape[2:]

            first_subdir = False

    return num_images, num_patches, image_shape, patch_shape, num_heatmaps_per_patch


def get_histogram_for_dof_values(subdirs):
    dof_values = np.zeros((num_images, 4))

    current = 0
    for subdir in subdirs:

        in_dataset_fullpath = INPUT_DIRECTORY + '/' + subdir + "/rgbd_and_labels.h5"
        print in_dataset_fullpath
        in_dataset = h5py.File(in_dataset_fullpath)

        for i in range(in_dataset['dof_values'].shape[0]):
            dof_values[current] = np.copy(in_dataset['dof_values'][i])
            current += 1

    hist_list = []
    bin_edges_list = []

    for i in range(in_dataset['dof_values'].shape[1]):
        hist, bin_edges = np.histogram(in_dataset['dof_values'][:, i], NUM_BINS, (0, math.pi))
        hist_list.append(hist)
        bin_edges_list.append(bin_edges)

    return hist_list, bin_edges_list


def init_out_dataset():

    #determine the size of each dataset
    patches_dataset_size = [num_patches*NUM_VC_OUT] + list(patch_shape)
    images_dataset_size = [num_images] + list(image_shape)
    uvd_dataset_size = [num_images, NUM_VC_OUT, 3]
    patch_labels_dataset_size = [num_patches*NUM_VC_OUT, NUM_VC_OUT*math.pow(NUM_BINS, NUM_DOF)*NUM_DEPTH_BINS]
    dof_values_dataset_size = [num_images, NUM_DOF]
    palm_to_object_offset_dataset_size = [num_images, 1]

    #determine the size of a chunk for each dataset
    patches_chunk_size = tuple([10] + list(patch_shape))
    images_chunk_size = tuple([10] + list(image_shape))
    uvd_chunk_size = (10, NUM_VC_OUT, 3)
    patch_labels_chunk_size = tuple([10, NUM_VC_OUT*math.pow(NUM_BINS, NUM_DOF)*NUM_DEPTH_BINS])
    dof_values_chunk_size = (1000, NUM_DOF)
    palm_to_object_offset_chunk_size = (1000, 1)

    #initialize the datasets
    out_dataset = h5py.File("out.h5")
    out_dataset.create_dataset("rgbd_patches",  patches_dataset_size, chunks=patches_chunk_size)
    out_dataset.create_dataset("rgbd",  images_dataset_size, chunks=images_chunk_size)
    out_dataset.create_dataset("rgbd_patch_labels",  patch_labels_dataset_size, chunks=patch_labels_chunk_size)
    out_dataset.create_dataset("dof_values", dof_values_dataset_size, chunks=dof_values_chunk_size)
    out_dataset.create_dataset("uvd", uvd_dataset_size, chunks=uvd_chunk_size)
    out_dataset.create_dataset("palm_to_object_offset", palm_to_object_offset_dataset_size, chunks=palm_to_object_offset_chunk_size )
    out_dataset.create_dataset("image_id", (num_patches*NUM_VC_OUT, 1), chunks=(1000, 1))

    return out_dataset


if __name__ == '__main__':

    subdirs = os.listdir(INPUT_DIRECTORY)

    #quickly run through the input directory to determine the values for several variables
    num_images, num_patches, image_shape, patch_shape, num_heatmaps_per_patch = get_data_dimensions(subdirs)

    #run through the dof values for all the grasps to determine the number of different
    #grasp categories.
    hist_list, bin_edges_list = get_histogram_for_dof_values(subdirs)

    #initialize the h5 dataset we are going to create
    out_dataset = init_out_dataset()

    uvd_selector = np.zeros(NUM_VC_IN)
    for index in VC_INDICES:
        uvd_selector[index] = 1


    print "need to eliminate CAMERA_BACKOFF_DISTANCE at some point"

    image_count = 0
    patch_count = 0
    for subdir in subdirs:

        in_dataset_fullpath = INPUT_DIRECTORY + '/' + subdir + "/rgbd_and_labels.h5"
        in_dataset = h5py.File(in_dataset_fullpath)

        print in_dataset_fullpath

        for i in range(in_dataset['rgbd_patches'].shape[0]):

            out_dataset['rgbd'][image_count] = in_dataset['rgbd'][i]
            out_dataset['uvd'][image_count] = in_dataset['uvd'][i][uvd_selector > 0]
            out_dataset['dof_values'][image_count] = in_dataset['dof_values'][i]

            u, v, d = in_dataset['uvd'][i][PALM_INDEX]
            out_dataset['palm_to_object_offset'][image_count] = in_dataset['rgbd'][i, u, v, 3] - d

            for j in range(num_heatmaps_per_patch):
                if j in VC_INDICES:
                    out_dataset['rgbd_patches'][patch_count] = in_dataset['rgbd_patches'][i, j]
                    out_dataset['rgbd_patch_labels'][patch_count, get_label_index(in_dataset['dof_values'][i], bin_edges_list, VC_INDICES.index(j), d)] = 1
                    out_dataset['image_id'][patch_count] = image_count
                    patch_count += 1

            image_count += 1

    import IPython
    IPython.embed()