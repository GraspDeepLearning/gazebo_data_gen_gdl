import h5py
import os
import numpy as np
import math
import matplotlib.pyplot as plt

#the number of virtual contacts collected
NUM_VC_IN = 8

#the number of virtual contacts we actually want to keep.
#most likely just palm and fingertips
NUM_VC_OUT = 4

#this are the indices for the virtual contacts we want to keep
PALM_INDEX = 0
# for 16 vc grasps
# FINGER_1_INDEX = 8
# FINGER_2_INDEX = 12
# FINGER_3_INDEX = 16

#for 7 vc grasps
FINGER_1_INDEX = 5
FINGER_2_INDEX = 6
FINGER_3_INDEX = 7
VC_INDICES = [PALM_INDEX, FINGER_1_INDEX, FINGER_2_INDEX, FINGER_3_INDEX]

#number of degrees of freedom for the hand
NUM_DOF = 4

#this is the number of bins for each dof of the barret hand.  These are used to classify the grasp type
NUM_DOF_BINS = 4

#two bins, for touching or not touching palm to object
NUM_DEPTH_BINS = 2

#the root data directory
GDL_DATA_PATH = os.environ["GDL_PATH"] + "/data"

#the directory we are going to pull all the h5 files from.
INPUT_DIRECTORY = GDL_DATA_PATH + '/rgbd_images/11_20_16_46'


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
        hist, bin_edges = np.histogram(in_dataset['dof_values'][:, i], NUM_DOF_BINS, (0, math.pi))
        hist_list.append(hist)
        bin_edges_list.append(bin_edges)

    return hist_list, bin_edges_list


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
    assert bin_id < NUM_DOF_BINS

    return bin_id


def get_num_grasp_types():
    return math.pow(NUM_DOF_BINS, NUM_DOF) * NUM_DEPTH_BINS


def get_grasp_type(dof_values, dof_bin_edges_list, d):

    grasp_type = 0

    for i in range(NUM_DOF):

        dof_value = dof_values[i]
        bin_edges = dof_bin_edges_list[i]

        #this should be between 0 and 3 inclusive
        bin_id = get_bin(dof_value, bin_edges)

        grasp_type += bin_id * math.pow(NUM_DOF_BINS, NUM_DOF_BINS-i-1)

    d_bin = 0
    if d > .02:
        d_bin = 1

    grasp_type += d_bin * math.pow(NUM_DOF_BINS, NUM_DOF)

    return grasp_type


def init_out_dataset():

    #determine the size of each dataset
    patches_dataset_size = [num_patches*NUM_VC_OUT] + list(patch_shape)
    images_dataset_size = [num_images] + list(image_shape)
    uvd_dataset_size = [num_images, NUM_VC_OUT, 3]
    patch_labels_dataset_size = [num_patches*NUM_VC_OUT, num_labels]
    dof_values_dataset_size = [num_images, NUM_DOF]
    palm_to_object_offset_dataset_size = [num_images, 1]

    #determine the size of a chunk for each dataset
    patches_chunk_size = tuple([10] + list(patch_shape))
    images_chunk_size = tuple([10] + list(image_shape))
    uvd_chunk_size = (10, NUM_VC_OUT, 3)
    patch_labels_chunk_size = tuple([10, num_labels])
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

    #the id of the image that this patch comes from
    out_dataset.create_dataset("patch_image_id", (num_patches*NUM_VC_OUT, 1), chunks=(1000, 1))
    #the id of the grasp type that this patch is for
    out_dataset.create_dataset("patch_grasp_type_id", (num_patches*NUM_VC_OUT, 1))
    #the id of the virtual contact that this patch is for
    out_dataset.create_dataset("patch_vc_id", (num_patches*NUM_VC_OUT, 1))

    return out_dataset


def build_dataset():
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

            #the id of the grasp type
            grasp_type_id = get_grasp_type(in_dataset['dof_values'][i], bin_edges_list, d)

            for j in range(num_heatmaps_per_patch):
                if j in VC_INDICES:

                    #the id of the virtual contact
                    vc_id = VC_INDICES.index(j)

                    #the label for the specific patch which is unique to both grasp type and virtual contact
                    grasp_full_label = vc_id + grasp_type_id*NUM_VC_OUT

                    out_dataset['rgbd_patches'][patch_count] = in_dataset['rgbd_patches'][i, j]
                    out_dataset['rgbd_patch_labels'][patch_count, grasp_full_label] = 1
                    out_dataset['patch_image_id'][patch_count] = image_count
                    out_dataset['patch_grasp_type_id'] = grasp_type_id
                    out_dataset['patch_vc_id'][patch_count] = vc_id
                    patch_count += 1

            image_count += 1


#now we are going to condense the dataset to only include grasps that have a reasonably large number of examples
#this will remove lots of the labels that do not actually correspond to feasible grasps.
def condense_dataset():

    grasp_types = out_dataset['patch_grasp_type_id'][:]

    counts = np.zeros(num_grasp_types + 1)
    for grasp_type_id in grasp_types:
            counts[grasp_type_id] += 1

    count_mask = counts > 10

    #this is the number of grasps that we have more than 10 training examples of
    num_condensed_grasp_types = sum(count_mask)
    num_condensed_labels = num_condensed_grasp_types*NUM_VC_OUT

    #we are getting rid of patches that belong to grasps that we have less than 10 training examples for
    #we are now going to figure out how many good patches we still have and also a mapping from label to condensed label.
    num_condensed_patches = 0
    grasp_type_to_condensed_grasp_type = {}
    for grasp_type in grasp_types:
        if count_mask[grasp_type]:
            grasp_type_to_condensed_grasp_type[grasp_type] = num_condensed_patches
            num_condensed_patches += 1

    condensed_dataset = h5py.File("out_condensed.h5")
    condensed_dataset.create_dataset('rgbd_patches', (num_condensed_patches, image_shape[0], image_shape[1], image_shape[2]), chunks=(10, image_shape[0], image_shape[1], image_shape[2]))
    condensed_dataset.create_dataset('rgbd_patch_labels', (num_condensed_patches, num_condensed_labels), chunks=(1000, num_condensed_labels))

    current_index = 0
    for i in range(grasp_types.shape[0]):
        grasp_type = grasp_types[i]
        vc_id = out_dataset['patch_vc_id'][i]

        if grasp_type in grasp_type_to_condensed_grasp_type:

            condensed_grasp_type = grasp_type_to_condensed_grasp_type[grasp_type]
            label = condensed_grasp_type*NUM_VC_OUT + vc_id
            label_array = np.zeros(num_condensed_labels)
            label_array[label] = 1
            condense_dataset['rgbd_patches'][current_index] = out_dataset['rgbd_patches'][i]
            condense_dataset['rgbd_patches'][current_index] = label_array

            current_index += 1


if __name__ == '__main__':

    subdirs = os.listdir(INPUT_DIRECTORY)

    #quickly run through the input directory to determine the values for several variables
    num_images, num_patches, image_shape, patch_shape, num_heatmaps_per_patch = get_data_dimensions(subdirs)

    num_grasp_types = get_num_grasp_types()
    num_labels = num_grasp_types*NUM_VC_OUT

    #run through the dof values for all the grasps to determine the number of different
    #grasp categories.
    hist_list, bin_edges_list = get_histogram_for_dof_values(subdirs)

    #initialize the h5 dataset we are going to create
    out_dataset = init_out_dataset()

    uvd_selector = np.zeros(NUM_VC_IN)
    for index in VC_INDICES:
        uvd_selector[index] = 1

    build_dataset()
    condense_dataset()



    import IPython
    IPython.embed()