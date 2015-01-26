import h5py
import yaml
from collections import namedtuple


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

    def add_grasp(self, grasp):

        current_limit = self.dset['rgbd'].shape[0]
        current_index = self.get_current_index()
        if current_index < current_limit:
            for key in self.config.keys():
                self.dset[key].resize(current_limit + 50, axis=0)

        for key in self.config.keys():
            self.dset[key][current_index] = grasp[key]

        self.increment_current_index()

    def get_grasp(self, index):

        grasp_dict = {}

        for key in self.config.keys():
            grasp_dict[key] = self.dset["rgbd"][index]

        return self.Grasp(**grasp_dict)

    def get_current_index(self):
        return self.dset['current_grasp_index'][0]

    def increment_current_index(self):
        self.dset['current_grasp_index'][0] += 1

    def iterator(self):
        return GraspIterator(self)


class GraspIterator():

    def __init__(self, dataset):
        self.dataset = dataset
        self.current_index = 0

    def __iter__(self):
        return self

    def next(self):

        #we have iterated over all the grasps
        if self.current_index >= self.dataset.get_current_index():
            raise StopIteration()

        grasp = self.dataset.get_grasp(self.current_index)
        self.current_index += 1

        return grasp


