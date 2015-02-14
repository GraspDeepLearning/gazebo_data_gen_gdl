import h5py
import os
import pickle
import numpy as np
from collections import namedtuple

from grasp_priors import GraspPriorsList, GraspPrior
from choose import choose_from, choose_from_or_none
from paths import CONDENSED_GAZEBO_DIR, GRASP_PRIORS_DIR


class GraspPriorListGen():

    def __init__(self, in_filepath='out.h5', out_filepath="grasp_priors_list.pkl"):
        self.in_filepath = in_filepath
        self.out_filepath = out_filepath

    def run(self):

        dset = h5py.File(self.in_filepath)

        wrist_roll = dset['wrist_roll']
        uvd = dset['uvd']
        joint_values = dset['joint_values']

        num_grasp_types = np.max(dset['grasp_type']) + 1
        grasp_types_count = np.zeros(num_grasp_types)

        avg_wrist_roll= np.zeros((num_grasp_types, 1))
        avg_uvd = np.zeros((num_grasp_types, 17, 3))
        avg_joint_values = np.zeros((num_grasp_types, 8))

        for i in range(len(dset["grasp_type"])):
            grasp_type = dset["grasp_type"][i][0]
            grasp_types_count[grasp_type] += 1

            avg_wrist_roll[grasp_type] += wrist_roll[i]
            avg_uvd[grasp_type] += uvd[i]
            avg_joint_values[grasp_type] += joint_values[i]

        for i in range(len(grasp_types_count)):
            avg_wrist_roll[i]/=grasp_types_count[i]
            avg_uvd[i]/=grasp_types_count[i]
            avg_joint_values[i]/=grasp_types_count[i]

        grasp_priors_list = GraspPriorsList()

        for i in range(len(grasp_types_count)):
            grasp_prior = GraspPrior()

            grasp_prior.wrist_roll = avg_wrist_roll[i]
            grasp_prior.uvd = avg_uvd[i]
            grasp_prior.joint_values = avg_joint_values[i]

            grasp_priors_list.add_grasp_prior(grasp_prior)

        f = open(self.out_filepath, 'w')
        pickle.dump(grasp_priors_list, f)
        f.close()



if __name__ == "__main__":

    # Choose in and out files.
    in_file = choose_from(CONDENSED_GAZEBO_DIR)
    in_filepath = CONDENSED_GAZEBO_DIR + in_file

    out_path = GRASP_PRIORS_DIR + in_file[:-3] + "/"
    try:
        os.stat(out_path)
    except:
        os.mkdir(out_path)

    out_filepath = out_path + "grasp_priors_list.pkl"

    gpl_generator = GraspPriorListGen(in_filepath=in_filepath, out_filepath=out_filepath)
    gpl_generator.run()

