
import os
import rospkg


rospack = rospkg.RosPack()

DATASET_TEMPLATE_PATH = rospack.get_path('grasp_dataset')

RAW_GRASPIT_DIR = os.path.expanduser("~/grasp_deep_learning/data/grasp_datasets/0_raw_graspit/")
AGG_GRASPIT_DIR = os.path.expanduser("~/grasp_deep_learning/data/grasp_datasets/1_agg_graspit/")
RAW_GAZEBO_DIR = os.path.expanduser("~/grasp_deep_learning/data/grasp_datasets/2_raw_gazebo/")
CONDENSED_GAZEBO_DIR = os.path.expanduser("~/grasp_deep_learning/data/grasp_datasets/3_condensed_gazebo/")
RAW_PYLEARN_DIR = os.path.expanduser("~/grasp_deep_learning/data/grasp_datasets/4_raw_pylearn/")
GRASP_PRIORS_DIR = os.path.expanduser("~/grasp_deep_learning/data/grasp_datasets/5_grasp_priors/")
