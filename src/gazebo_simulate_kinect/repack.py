import os
import h5py

path = "/home/jared/grasp_deep_learning/data/rgbd_images/11_20_16_46/"
subdirs = os.listdir(path)

for subdir in subdirs:
    in_dataset_path = path + subdir + '/rgbd_and_labels.h5'
    out_dataset_path = path + subdir + '/no_labels.h5'

    print in_dataset_path
    print out_dataset_path

    in_dataset = h5py.File(path + subdir + '/rgbd_and_labels.h5')
    out_dataset = h5py.File(path + subdir + '/no_labels.h5')

    out_dataset.create_dataset("rgbd", (in_dataset['rgbd'].shape[0], 480, 640, 4), chunks=(10, 480, 640, 4))
    out_dataset.create_dataset("rgbd_patches", (in_dataset['rgbd_patches'].shape[0], 8, 72, 72, 4), chunks=(10, 8, 72, 72, 4))
    out_dataset.create_dataset("rgbd_patch_labels", (in_dataset['rgbd_patch_labels'].shape[0], 1))
    out_dataset.create_dataset("dof_values", (in_dataset['dof_values'].shape[0], 4), chunks=(100, 4))
    out_dataset.create_dataset("uvd", (in_dataset['uvd'].shape[0], 8, 3), chunks=(100, 8, 3))

    for i in range(in_dataset['rgbd'].shape[0]):
        for key in out_dataset.keys():
            out_dataset[key][i] = in_dataset[key][i]