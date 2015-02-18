
import os
import time
import rospkg

from choose import choose_from
from grasp import get_model_grasps
from grasp_dataset import GraspDataset

def get_date_string():
    t = time.localtime()
    minute = str(t.tm_min)
    if len(minute) == 1:
        minute = '0' + minute
    t_string = str(t.tm_mon) + "_" + str(t.tm_mday) + "_" + str(t.tm_hour) + "_" + minute
    return t_string

if __name__ == "__main__":

    graspit_dataset_dir = os.path.expanduser("~/grasp_deep_learning/data/grasp_datasets/0_raw_graspit/")
    graspit_dir = choose_from(graspit_dataset_dir)
    graspit_grasps_dir = (graspit_dataset_dir + graspit_dir) + '/'

    graspit_agg_dir = os.path.expanduser("~/grasp_deep_learning/data/grasp_datasets/1_agg_graspit/")

    rospack = rospkg.RosPack()
    DATASET_TEMPLATE_PATH = rospack.get_path('grasp_dataset')

    grasp_dataset = GraspDataset(graspit_agg_dir + graspit_dir + "-custom-" + get_date_string() + ".h5",
                                 DATASET_TEMPLATE_PATH + "/dataset_configs/graspit_grasps_dataset.yaml")

    grasps = []
    print "reading grasps from files"
    for model_name in os.listdir(graspit_grasps_dir):
        grasps = grasps + get_model_grasps(graspit_grasps_dir + model_name, model_name, graspClass=grasp_dataset.Grasp)

    print "writing low energy grasps to h5"
    num = 0
    listSize = len(grasps)

    for grasp in grasps:
        if num % 1000 is 0:
            print "%s / %s" % (num, listSize)

        num += 1
        try:
            grasp_dataset.add_grasp(grasp)
        except:
            import IPython; IPython.embed()




    import IPython
    IPython.embed()
    assert False
