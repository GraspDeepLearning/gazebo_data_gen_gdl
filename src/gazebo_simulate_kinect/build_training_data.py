#!/usr/bin/env python
import rospy
import rospkg

from geometry_msgs.msg import Pose
import std_srvs.srv

import numpy as np
import os
from time import sleep
import math
import tf
import random
from scipy import misc
import h5py

import matplotlib.pyplot as plt

from src.gazebo_model_manager import GazeboKinectManager, GazeboModelManager
from src.grasp import get_model_grasps
from src.transformer_manager import TransformerManager
from src.xyz_to_pixel_loc import xyz_to_uv

rospack = rospkg.RosPack()

GAZEBO_MODEL_PATH = os.environ["GAZEBO_MODEL_PATH"]
GRASPABLE_MODEL_PATH = GAZEBO_MODEL_PATH
#GRASPABLE_MODEL_PATH = rospack.get_path('object_models') + "/models/cgdb/model_database/"

def get_model_orientations():
    model_orientations = []
    for r in np.linspace(0, 2*math.pi, num=10):
        for p in np.linspace(0, 2*math.pi, num=10):
            for y in np.linspace(0, 2*math.pi, num=5):
                model_orientations.append((r, p, y))
    return model_orientations


def gen_model_pose(model_orientation):
    model_pose = Pose()
    model_pose.position.x = 2 + random.uniform(-.25, .25)
    model_pose.position.y = 0.0 + random.uniform(-.25, .25)
    model_pose.position.z = 1 + random.uniform(-.25, .25)

    roll = model_orientation[0]
    pitch = model_orientation[1]
    yaw = model_orientation[2]

    quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
    model_pose.orientation.x = quaternion[0]
    model_pose.orientation.y = quaternion[1]
    model_pose.orientation.z = quaternion[2]
    model_pose.orientation.w = quaternion[3]

    return model_pose

if __name__ == '__main__':

    output_image_dir = os.path.expanduser("~/grasp_deep_learning/data/rgbd_images/")
    models_dir = GRASPABLE_MODEL_PATH

    kinect_manager = GazeboKinectManager()
    kinect_manager.spawn_kinect()

    pause_physics_service_proxy = rospy.ServiceProxy("/gazebo/pause_physics", std_srvs.srv.Empty)
    unpause_physics_service_proxy = rospy.ServiceProxy("/gazebo/unpause_physics", std_srvs.srv.Empty)
    #we don't need any physics right now
    pause_physics_service_proxy()

    model_manager = GazeboModelManager(models_dir=models_dir)

    model_orientations = get_model_orientations()

    for model_name in os.listdir(os.path.expanduser("~/grasp_deep_learning/data/grasps/")):

        model_output_image_dir = output_image_dir + model_name + '/'
        if not os.path.exists(model_output_image_dir):
            os.makedirs(model_output_image_dir)

        model_type = model_name
        model_manager.spawn_model(model_name, model_type)

        transform_manager = TransformerManager()

        camera_pose_in_world_frame = model_manager.get_model_state(kinect_manager.camera_name).pose
        transform_manager.add_transform(camera_pose_in_world_frame, "World", "Camera")

        dataset = h5py.File(model_output_image_dir + "rgbd_and_labels.h5")
        num_images = len(model_orientations)

        dataset.create_dataset("rgbd", (num_images, 480, 640, 4), chunks=(10, 480, 640, 4))
        dataset.create_dataset("labels", (num_images, 480, 640), chunks=(10, 480, 640))

        for index in range(len(model_orientations)):
            model_orientation = model_orientations[index]
            model_pose = gen_model_pose(model_orientation)

            model_manager.set_model_state(model_name, model_pose)

            sleep(1)

            rgbd_image = np.copy(kinect_manager.get_rgbd_image())

            grasp_points = np.zeros((480, 640))
            overlay = np.copy(rgbd_image[:, :, 0])

            model_pose_in_world_frame = model_manager.get_model_state(model_name).pose
            transform_manager.add_transform(model_pose_in_world_frame, "World", "Model")

            model_grasps = get_model_grasps(model_name)

            for model_grasp in model_grasps:

                transform_manager.add_transform(model_grasp.pose, "Model", "Grasp")

                #get grasp point in camera frame
                grasp_in_camera_frame = transform_manager.transform_pose(model_grasp.pose, "Grasp", "Camera")
                #grasp_in_camera_frame = transform_pose(model_grasp.pose, "Model", "Camera", transformer)
                grasp_in_camera_frame = grasp_in_camera_frame.pose

                #this is the pixel location of the grasp point
                u, v = xyz_to_uv((grasp_in_camera_frame.position.x, grasp_in_camera_frame.position.y, grasp_in_camera_frame.position.z))

                #import IPython
                #IPython.embed()
                if(u < overlay.shape[0]-2 and u > -2 and v < overlay.shape[1]-2 and v > -2):
                    overlay[u-2:u+2, v-2:v+2] = model_grasp.energy
                    grasp_points[u, v] = model_grasp.energy

            output_filepath = model_output_image_dir + model_name + "_" + str(index)
            if not os.path.exists(output_filepath):
                os.makedirs(output_filepath)

            #fix nans in depth
            max_depth = np.nan_to_num(rgbd_image[:, :, 3]).max()*1.3
            for x in range(rgbd_image.shape[0]):
                for y in range(rgbd_image.shape[1]):
                    if rgbd_image[x, y, 3] != rgbd_image[x, y, 3]:
                        rgbd_image[x, y, 3] = max_depth

            #normalize rgb:
            rgbd_image[:, :, 0:3] = rgbd_image[:, :, 0:3]/255.0
            #normalize d
            rgbd_image[:, :, 3] = rgbd_image[:, :, 3]/rgbd_image[:, :, 3].max()
            #normalize grasp_points
            #all nonzero grasp points are currently negative, so divide by the min.
            grasp_points = grasp_points/grasp_points.min()

            dataset["rgbd"][index] = rgbd_image
            dataset["labels"][index] = grasp_points

            misc.imsave(output_filepath + "/" + 'out.png', grasp_points)
            misc.imsave(output_filepath + "/" + 'overlay.png', overlay)
            misc.imsave(output_filepath + "/" + 'r.png', rgbd_image[:, :, 0])
            misc.imsave(output_filepath + "/" + 'g.png', rgbd_image[:, :, 1])
            misc.imsave(output_filepath + "/" + 'b.png', rgbd_image[:, :, 2])
            misc.imsave(output_filepath + "/" + 'd.png', rgbd_image[:, :, 3])

        model_manager.remove_model(model_name)
