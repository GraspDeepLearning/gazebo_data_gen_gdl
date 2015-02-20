
import os
import rospkg


rospack = rospkg.RosPack()

DATASET_TEMPLATE_PATH = rospack.get_path('grasp_dataset')

#DIR_PREFIX = "~/grasp_deep_learning/data/grasp_datasets/"
DIR_PREFIX = "/media/Elements/gdl_data/grasp_datasets/"

RAW_GRASPIT_DIR = os.path.expanduser(DIR_PREFIX + "0_raw_graspit/")
AGG_GRASPIT_DIR = os.path.expanduser(DIR_PREFIX + "1_agg_graspit/")
RAW_GAZEBO_DIR = os.path.expanduser(DIR_PREFIX + "2_raw_gazebo/")
CONDENSED_GAZEBO_DIR = os.path.expanduser(DIR_PREFIX + "3_condensed_gazebo/")
RAW_PYLEARN_DIR = os.path.expanduser(DIR_PREFIX + "4_raw_pylearn/")
GRASP_PRIORS_DIR = os.path.expanduser(DIR_PREFIX + "5_grasp_priors/")
