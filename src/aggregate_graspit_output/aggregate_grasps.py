
import os
import rospkg
from grasp import get_model_grasps
from grasp_dataset import GraspDataset

if __name__ == "__main__":

    graspit_grasps_dir = os.path.expanduser("~/grasp_deep_learning/data/grasp_datasets/contact_and_potential_grasps/")
    graspit_agg_dir = os.path.expanduser("~/grasp_deep_learning/data/grasp_datasets/")

    rospack = rospkg.RosPack()
    DATASET_TEMPLATE_PATH = rospack.get_path('grasp_dataset')

    grasp_dataset = GraspDataset(graspit_agg_dir + "contact_and_potential_grasps.h5",
                                 DATASET_TEMPLATE_PATH + "/dataset_configs/graspit_grasps_dataset.yaml")

    grasps = []
    print "reading grasps from files"
    for model_name in os.listdir(graspit_grasps_dir):
        grasps = grasps + get_model_grasps(graspit_grasps_dir + model_name, model_name, graspClass=grasp_dataset.Grasp)

    print "removing high energy grasps"
    for grasp in grasps:
        if grasp.energy > 0:
            grasps.remove(grasp)


    print "writing low energy grasps to h5"
    grasp_dataset.add_grasps(grasps)


    import IPython
    IPython.embed()
    assert False
