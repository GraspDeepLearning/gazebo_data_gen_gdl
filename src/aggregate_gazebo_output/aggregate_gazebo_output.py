import math
import numpy as np
import copy

from grasp_dataset import GraspDataset
from choose import choose_from
from paths import CONDENSED_GAZEBO_DIR, RAW_GAZEBO_DIR, DATASET_TEMPLATE_PATH

NUM_CONDENSED_GRASP_TYPES = 8
NUM_BINS_PER_JOINT = 5


#helper function to determine what bin a data point belongs in.
def get_bin(data_point, bin_edges):
    bin_id = -1
    for bin_edge in bin_edges:

        #this will never pass on first bin_edge
        if data_point < bin_edge:
            break

        bin_id += 1

    #sanity check
    assert bin_id >= 0
    assert bin_id < len(bin_edges) - 1

    return bin_id


def get_grasp_type(bin_values, bin_edges_list, num_entries_per_bin):

    grasp_type = 0

    for i in range(len(bin_values)):

        bin_value = bin_values[i]
        bin_edges = bin_edges_list[i]

        bin_id = get_bin(bin_value, bin_edges)

        grasp_type += bin_id * math.pow(num_entries_per_bin, i)

    return grasp_type


#now we are going to condense the dataset to only include grasps that have a reasonably large number of examples
#this will remove lots of the labels that do not actually correspond to feasible grasps.
def condense_grasp_types(grasp_types, num_grasp_types):

    # grasp_types_sorted = copy.copy(list(set(grasp_types)))
    # grasp_types_sorted.sort()
    # grasp_types_sorted.reverse()

    counts = np.zeros(num_grasp_types)
    for grasp_type_id in grasp_types:
        counts[grasp_type_id] += 1

    counts_sorted = list(counts)
    counts_sorted.sort()
    counts_sorted.reverse()

    threshold = counts_sorted[NUM_CONDENSED_GRASP_TYPES - 1]

    grasp_type_to_condensed_grasp_type = {}
    current_condensed_grasp_id = 0
    for i in range(len(counts)):
        if counts[i] >= threshold:
            grasp_type_to_condensed_grasp_type[i] = current_condensed_grasp_id
            current_condensed_grasp_id += 1

    return grasp_type_to_condensed_grasp_type


if __name__ == "__main__":

    bin_names = ['bhand/finger_1/prox_joint',
                 'bhand/finger_1/med_joint',
                 'bhand/finger_1/dist_joint',
                 'bhand/finger_2/prox_joint',
                 'bhand/finger_2/med_joint',
                 'bhand/finger_2/dist_joint',
                 'bhand/finger_3/med_joint',
                 'bhand/finger_3/dist_joint',
                 'wrist_roll']
    slop = .01
    bin_ranges = [(0,  math.pi + slop),
                  (0, 2.44 + slop),
                  (0, .84 + slop),
                  (0,  math.pi + slop),
                  (0, 2.44 + slop),
                  (0, .84 + slop),
                  (0, 2.44 + slop),
                  (0, .84 + slop),
                  (-math.pi - slop, math.pi + slop)]

    num_entries_per_bin = NUM_BINS_PER_JOINT

    num_grasp_types = math.pow(num_entries_per_bin, len(bin_ranges))

    bin_edges = []
    for bin_range in bin_ranges:
        hist, edges = np.histogram([], num_entries_per_bin, bin_range)
        bin_edges.append(edges)

    gazebo_raw_file = choose_from(RAW_GAZEBO_DIR)
    gazebo_grasp_dataset = GraspDataset(RAW_GAZEBO_DIR + gazebo_raw_file,
                                        DATASET_TEMPLATE_PATH + "/dataset_configs/gazebo_capture_config.yaml")

    count = 0

    grasp_types = []
    print "Iterating through raw grasps"
    for grasp in gazebo_grasp_dataset.iterator():

        if count % 500 == 0:
            print str(count) + '/' + str(gazebo_grasp_dataset.get_current_index())

        bins = list(grasp.joint_values)
        bins.append(grasp.wrist_roll[0])

        grasp_type = get_grasp_type(bins, bin_edges, num_entries_per_bin)
        grasp_types.append(int(grasp_type))

        count += 1

    print "Building grasp type to condensed grasp type dict"
    grasp_type_to_condensed_grasp_type = condense_grasp_types(grasp_types, num_grasp_types)

    actual_num_condensed_grasp_types = len(grasp_type_to_condensed_grasp_type.keys())
    condensed_gazebo_path = CONDENSED_GAZEBO_DIR + gazebo_raw_file[:-3] + "_" + str(actual_num_condensed_grasp_types) + "_grasp_types.h5"

    condensed_gazebo_grasp_dataset = GraspDataset(condensed_gazebo_path,
                                                  DATASET_TEMPLATE_PATH + "/dataset_configs/gazebo_condensed_config.yaml")

    print "Saving condensed grasps"
    for grasp in gazebo_grasp_dataset.iterator():
        bins = list(grasp.joint_values)
        bins.append(grasp.wrist_roll[0])
        grasp_type = get_grasp_type(bins, bin_edges, num_entries_per_bin)

        if grasp_type in grasp_type_to_condensed_grasp_type:
            condensed_grasp_type = grasp_type_to_condensed_grasp_type[grasp_type]

            condensed_grasp = condensed_gazebo_grasp_dataset.Grasp(rgbd=grasp.rgbd,
                                                                   dof_values=grasp.dof_values,
                                                                   palm_pose=grasp.palm_pose,
                                                                   joint_values=grasp.joint_values,
                                                                   uvd=grasp.uvd,
                                                                   wrist_roll=grasp.wrist_roll,
                                                                   virtual_contacts=grasp.virtual_contacts,
                                                                   model_name=grasp.model_name,
                                                                   energy=grasp.energy,
                                                                   grasp_type=condensed_grasp_type)

            condensed_gazebo_grasp_dataset.add_grasp(condensed_grasp)

    import IPython
    IPython.embed()










