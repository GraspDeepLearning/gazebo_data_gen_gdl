from grasp_dataset import GraspDataset

import rospkg
rospack = rospkg.RosPack()

DATASET_PATH = rospack.get_path('grasp_dataset')


grasp_dataset = GraspDataset()