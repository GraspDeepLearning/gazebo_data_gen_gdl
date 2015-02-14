import h5py
import yaml
from collections import namedtuple
import random
import numpy as np

class GraspDataset():

    def __init__(self, dset_full_filepath, dset_config_full_filepath=None):
        self.dset_full_filepath = dset_full_filepath
        self.dset_config_full_filepath = dset_config_full_filepath

        self.config = yaml.load(open(dset_config_full_filepath))
        self.dset = h5py.File(self.dset_full_filepath)

        #this named tuple has a field for every field in the config file
        self.Grasp = namedtuple('Grasp', self.config.keys())

        for dset_key in self.config.keys():
            if not dset_key in self.dset.keys():
                table_config = self.config[dset_key]
                if 'dtype' in table_config and table_config['dtype'] == 'String':
                    dt = h5py.special_dtype(vlen=bytes)
                    self.dset.create_dataset(name=dset_key,
                                             shape=tuple(table_config['chunk_shape']),
                                             maxshape=tuple(table_config['max_shape']),
                                             chunks=tuple(table_config['chunk_shape']),
                                             dtype=dt)
                else:
                    self.dset.create_dataset(name=dset_key,
                                             shape=tuple(table_config['chunk_shape']),
                                             maxshape=tuple(table_config['max_shape']),
                                             chunks=tuple(table_config['chunk_shape']))

        #this is the index of the next grasp location
        if not 'current_grasp_index' in self.dset.keys():
            self.dset.create_dataset(name='current_grasp_index', shape=(1,))
            self.dset['current_grasp_index'][0] = 0

    #this method will do just one resize, rather
    #than doing so at every nth insertion.  This should
    #be much faster.
    def add_grasps(self, grasps):
        current_limit = self.dset[self.config.keys()[0]].shape[0]
        current_index = self.get_current_index()
        if current_index + len(grasps) > current_limit:
            difference = current_index + len(grasps) - current_limit
            for key in self.config.keys():
                self.dset[key].resize(current_limit + difference, axis=0)

        for grasp in grasps:
            self.add_grasp(grasp)

    # adds a single entry to the dataset
    def add_grasp(self, grasp):

        current_limit = self.dset[self.config.keys()[0]].shape[0]
        current_index = self.get_current_index()
        if current_index >= current_limit:
            for key in self.config.keys():
                self.dset[key].resize(current_limit + 250, axis=0)

        for key in self.config.keys():
            self.dset[key][current_index] = grasp.__getattribute__(key)

        self.increment_current_index()


    def get_grasp(self, index):

        grasp_dict = {}

        for key in self.config.keys():
            grasp_dict[key] = self.dset[key][index]

        return self.Grasp(**grasp_dict)

    def get_current_index(self):
        return self.dset['current_grasp_index'][0]

    def increment_current_index(self):
        self.dset['current_grasp_index'][0] += 1

    def iterator(self, start=None, end=None):
        return GraspIterator(self, start, end)

    def random_iterator(self, num_items=None):
        return RandomGraspIterator(self, num_items)


class GraspIterator():

    def __init__(self, dataset, start=None, end=None):
        self.dataset = dataset
        self.current_index = 0

        if end:
            self.end_index = end
        else:
            self.end_index = self.dataset.get_current_index()

        if start:
            self.current_index = start

    def __iter__(self):
        return self

    def next(self):

        #we have iterated over all the grasps
        if self.current_index >= self.end_index:
            raise StopIteration()

        grasp = self.dataset.get_grasp(self.current_index)
        self.current_index += 1

        return grasp


class RandomGraspIterator():

    def __init__(self, dataset, num_items=None):
        self.dataset = dataset

        if num_items is None:
            num_items = self.dataset.get_current_index()

        self.order = np.random.permutation(int(self.dataset.get_current_index()))

        self.num_items = num_items

        self.current_index = 0

    def __iter__(self):
        return self

    def next(self):
        #we have iterated over all the grasps
        if self.current_index >= self.num_items:
            raise StopIteration()

        self.current_index += 1

        return self.dataset.get_grasp(self.order[self.current_index])


