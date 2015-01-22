import h5py
import numpy as np

from collections import namedtuple
import pickle





from grasp_priors import GraspPriorsList, GraspPrior

class GraspPriorListGen():

    def __init__(self, in_filename='out_condensed', out_filename="grasp_priors_list.pkl"):
        self.in_filename = in_filename
        self.out_filename = out_filename

    def run(self):


        dset = h5py.File(self.in_filename)

        dof_values = dset['dof_values']
        wrist_roll = dset['wrist_roll']
        uvd = dset['uvd']
        joint_values = dset['joint_values']
        palm_backoff = dset['palm_to_object_offset']

        num_grasp_types = np.max(dset['grasp_type_id']) + 1
        grasp_types_count = np.zeros(num_grasp_types)

        avg_dof = np.zeros((num_grasp_types, 4))
        avg_wrist_roll= np.zeros((num_grasp_types, 1))
        avg_uvd = np.zeros((num_grasp_types, 4, 3))
        avg_joint_values = np.zeros((num_grasp_types, 8))
        avg_palm_backoff = np.zeros((num_grasp_types, 1))

        for i in range(len(dset["grasp_type_id"])):

            grasp_type = dset["grasp_type_id"][i][0]
            grasp_types_count[grasp_type] += 1

            avg_dof[grasp_type] += dof_values[i]
            avg_wrist_roll[grasp_type] += wrist_roll[i]
            avg_uvd[grasp_type] += uvd[i]
            avg_joint_values[grasp_type] += joint_values[i]
            avg_palm_backoff[grasp_type] += palm_backoff[i]

        for i in range(len(grasp_types_count)):
            avg_dof[i]/=grasp_types_count[i]
            avg_wrist_roll[i]/=grasp_types_count[i]
            avg_uvd[i]/=grasp_types_count[i]
            avg_joint_values[i]/=grasp_types_count[i]
            avg_palm_backoff[i]/=grasp_types_count[i]

        grasp_priors_list = GraspPriorsList()

        for i in range(len(grasp_types_count)):
            grasp_prior = GraspPrior()

            grasp_prior.dof_values = avg_dof[i]
            grasp_prior.wrist_roll = avg_wrist_roll[i]
            grasp_prior.uvd = avg_uvd[i]
            grasp_prior.joint_values = avg_joint_values[i]
            grasp_prior.palm_backoff = avg_palm_backoff[i]

            grasp_priors_list.add_grasp_prior(grasp_prior)

        f = open(self.out_filename, 'w')
        pickle.dump(grasp_priors_list, f)
        f.close()



if __name__ == "__main__":

    gpl_generator = GraspPriorListGen()
    gpl_generator.run()

