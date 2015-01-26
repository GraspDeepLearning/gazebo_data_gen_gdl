#!/usr/bin/env python

import os
import rospkg
from grasp import get_model_grasps
from grasp_dataset import GraspDataset
from condense_via_energy import CondenseGraspsViaEnergy
#from condense_grasp_types import CondenseGraspTypes

if __name__ == "__main__":

    rospack = rospkg.RosPack()

    DATASET_TEMPLATE_PATH = rospack.get_path('grasp_dataset')
    GDL_GRASPS_PATH = os.environ["GDL_GRASPS_PATH"]
    GRASP_DATASET_PATH = os.environ["GDL_DATA_PATH"] + "/grasp_datasets/"
    if not os.path.exists(GRASP_DATASET_PATH):
        os.mkdir(GRASP_DATASET_PATH)

    grasp_dataset = GraspDataset(GRASP_DATASET_PATH + "dataset.h5",
                                 DATASET_TEMPLATE_PATH + "/dataset_configs/graspit_grasps_dataset.yaml")

    aggregated_grasps = {}

    for model_name in os.listdir(GDL_GRASPS_PATH):
        aggregated_grasps.setdefault(model_name, [])
        aggregated_grasps[model_name].extend(get_model_grasps(model_name, graspClass=grasp_dataset.Grasp))
        # Grasps are named tuples : 'energy joint_angles dof_values pose virtual_contacts'

    # Condense grasps via energy. Use automatic threshold
    aggregated_grasps = CondenseGraspsViaEnergy().run(aggregated_grasps)

    # Condense grasps via limiting grasp types. Use threshold = 400 (from old threshold)
    #aggregated_grasps = CondenseGraspTypes(threshold=400,
    #                                       num_dof=4,
    #                                       num_dof_bins=4,
    #                                       num_depth_bins=2).run(aggregated_grasps)


    grasp_dataset.add_grasp(aggregated_grasps for grasp in aggregated_grasps)