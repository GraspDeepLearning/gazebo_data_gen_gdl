import math
import numpy as np
from grasp_dataset import GraspDataset
import os
import rospkg
from paths import CONDENSED_GAZEBO_DIR, DATASET_TEMPLATE_PATH, RAW_PYLEARN_DIR
import h5py

PATCH_SIZE = 72
VC_INDICES = [1, 8, 12, 16]


def extract_patch(vc_uvd, rgbd_image):

    vc_u, vc_v, vc_d = vc_uvd
    return rgbd_image[vc_u-PATCH_SIZE/2:vc_u+PATCH_SIZE/2, vc_v-PATCH_SIZE/2:vc_v+PATCH_SIZE/2, :]


if __name__ == "__main__":

    gazebo_condensed_file = choose_from(CONDENSED_GAZEBO_DIR)[:-3]

    condensed_gazebo_grasp_dataset = GraspDataset(CONDENSED_GAZEBO_DIR + gazebo_condensed_file + ".h5",
                                        DATASET_TEMPLATE_PATH + "/dataset_configs/gazebo_condensed_config.yaml")

    dset = h5py.File(RAW_PYLEARN_DIR + gazebo_condensed_file + str(PATCH_SIZE) + ".h5")

    num_grasps = condensed_gazebo_grasp_dataset.get_current_index()
    num_grasp_types = condensed_gazebo_grasp_dataset.dset['grasp_type'][:].max() + 1
    num_labels = num_grasp_types*4

    dset.create_dataset("rgbd_patches", shape=(num_grasps*4, PATCH_SIZE, PATCH_SIZE, 4), chunks=(10, PATCH_SIZE, PATCH_SIZE, 4))
    dset.create_dataset("rgbd_patch_labels", shape=(num_grasps*4, num_labels), chunks=(10, num_labels))

    for i in range(num_grasps):
        grasp = condensed_gazebo_grasp_dataset.get_grasp(i)

        for j in range(4):
            vc_indice = VC_INDICES[j]
            uvd = grasp.uvd[vc_indice]

            patch = extract_patch(uvd, grasp.rgbd)
            label = grasp.grasp_type*4 + j

            dset['rgbd_patches'][i*4 + j] = patch
            dset['rgbd_patch_labels'][i*4+j, label] = grasp.energy














