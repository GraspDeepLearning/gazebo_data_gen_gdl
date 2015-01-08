import h5py
import numpy as np

dset = h5py.File("out_condensed.h5")

dof_values = dset['dof_values']
wrist_roll = dset['wrist_roll']
uvd = dset['uvd']
joint_values = dset['joint_values']


num_grasp_types = np.max(dset['grasp_type_id']) + 1
grasp_types_count = np.zeros(num_grasp_types)

avg_dof = np.zeros((num_grasp_types, 4))
avg_wrist_roll= np.zeros((num_grasp_types, 1))
avg_uvd = np.zeros((num_grasp_types, 4, 3))
avg_joint_values = np.zeros((num_grasp_types, 8))

for i in range(len(dset["grasp_type_id"])):

    grasp_type = dset["grasp_type_id"][i][0]
    grasp_types_count[grasp_type] += 1

    avg_dof[grasp_type] += dof_values[i]
    avg_wrist_roll[grasp_type] += wrist_roll[i]
    avg_uvd[grasp_type] += uvd[i]
    avg_joint_values[grasp_type] += joint_values[i]

for i in range(len(grasp_types_count)):
    avg_dof[i]/=grasp_types_count[i]
    avg_wrist_roll[i]/=grasp_types_count[i]
    avg_uvd[i]/=grasp_types_count[i]
    avg_joint_values[i]/=grasp_types_count[i]


import IPython
IPython.embed()
assert False
