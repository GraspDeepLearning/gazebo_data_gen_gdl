import random
import math
import unittest

import numpy as np

import aggregate_gazebo_output


class TestAggregateGazeboOutputFunctions(unittest.TestCase):

    def setUp(self):
        self.bin_names = ['bhand/finger_1/prox_joint',
                 'bhand/finger_1/med_joint',
                 'bhand/finger_1/dist_joint',
                 'bhand/finger_2/prox_joint',
                 'bhand/finger_2/med_joint',
                 'bhand/finger_2/dist_joint',
                 'bhand/finger_3/med_joint',
                 'bhand/finger_3/dist_joint',
                 'wrist_roll']

        self.bin_ranges = [(0, 3.14),
                  (0, 2.44),
                  (0, .84),
                  (0, 3.14),
                  (0, 2.44),
                  (0, .84),
                  (0, 2.44),
                  (0, .84),
                  (-math.pi, math.pi)]

        self.num_entries_per_bin = aggregate_gazebo_output.NUM_BINS_PER_JOINT

        self.num_grasp_types = math.pow(self.num_entries_per_bin, len(self.bin_ranges))

        self.bin_edges = []
        for bin_range in self.bin_ranges:
            hist, edges = np.histogram([], self.num_entries_per_bin, bin_range)
            self.bin_edges.append(edges)

    #show that the max grasp type goes to the correct bin
    def test_high_grasp_type(self):
        grasp = []
        for bin_range in self.bin_ranges:
            grasp.append(bin_range[1]-.001)

        grasp_type = aggregate_gazebo_output.get_grasp_type(grasp, self.bin_edges, self.num_entries_per_bin)
        self.assertEqual(grasp_type, self.num_grasp_types - 1)

    def test_low_grasp_type(self):
        grasp = []
        for bin_range in self.bin_ranges:
            grasp.append(bin_range[0]+.001)

        grasp_type = aggregate_gazebo_output.get_grasp_type(grasp, self.bin_edges, self.num_entries_per_bin)
        self.assertEqual(grasp_type, 0)



if __name__ == '__main__':
    unittest.main()