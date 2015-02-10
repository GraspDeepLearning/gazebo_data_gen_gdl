import math
import numpy as np


class CondenseGraspTypes():
    NUM_DEPTH_BINS = 2


    def __init__(self, threshold, num_dof, num_dof_bins, num_depth_bins):
        self.threshold = threshold
        self.num_dof = num_dof                  # number of degrees of freedom for the hand
        self.num_dof_bins = num_dof_bins        # this is the number of bins for each dof of the barret hand.
                                                # These are used to classify the grasp type
        self.num_depth_bins = num_depth_bins

    def get_num_grasp_types(self):
        wrist_roll_bins = np.arange(-math.pi, math.pi)
        return math.pow(self.num_dof_bins, self.num_dof) * self.num_depth_bins * len(wrist_roll_bins)

    #helper function to determine what bin a data point belongs in.
    def get_bin(self, data_point, bin_edges):
        bin_id = 0
        for bin_edge in bin_edges:

            #this will never pass on first bin_edge
            if data_point < bin_edge:
                break

            bin_id += 1

        bin_id -= 1

        #sanity check
        assert bin_id >= 0

        return bin_id

    def get_grasp_type(self, dof_values, dof_bin_edges_list, d, wrist_roll):

        grasp_type = 0

        for i in range(self.num_dof):

            dof_value = dof_values[i]
            bin_edges = dof_bin_edges_list[i]

            #this should be between 0 and 3 inclusive
            bin_id = self.get_bin(dof_value, bin_edges)

            grasp_type += bin_id * math.pow(self.num_dof_bins, self.num_dof_bins-i-1)

        d_bin = 0
        if d > .02:
            d_bin = 1


        wrist_roll_bins = np.arange(-math.pi, math.pi)
        wrist_roll_bin = self.get_bin(wrist_roll, wrist_roll_bins)


        grasp_type += d_bin * math.pow(self.num_dof_bins, self.num_dof)
        grasp_type += wrist_roll_bin * math.pow(self.num_dof_bins, self.num_dof) * self.num_depth_bins

        return grasp_type

    def run(self, grasps):
        """

        :param grasps:
        :return:
        """

        #the id of the grasp type
        grasp_type_id = self.get_grasp_type(grasp.dof_values, bin_edges_list, d, grasp.wrist_roll)

        return grasps
