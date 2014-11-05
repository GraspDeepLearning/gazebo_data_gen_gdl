#!/usr/bin/env python
import rospy
import rospkg

from geometry_msgs.msg import Pose
import std_srvs.srv

import numpy as np
import os
from time import sleep
import time
import math
import tf
import random
from scipy import misc
import h5py

from src.gazebo_model_manager import GazeboKinectManager, GazeboModelManager
from src.grasp import get_model_grasps
from src.transformer_manager import TransformerManager
from src.xyz_to_pixel_loc import xyz_to_uv

rospack = rospkg.RosPack()

GDL_OBJECT_PATH = os.environ["GDL_OBJECT_PATH"]
GRASPABLE_MODEL_PATH = GDL_OBJECT_PATH
GDL_DATA_PATH = os.environ["GDL_PATH"] + "/data"
GDL_GRASPS_PATH = os.environ["GDL_GRASPS_PATH"]


def gen_model_pose(model_orientation):
    model_pose = Pose()
    model_pose.position.x = 2 + random.uniform(-.25, .25)
    model_pose.position.y = 0.0 + random.uniform(-.25, .25)
    model_pose.position.z = 5 + random.uniform(-.25, .25)

    roll = model_orientation[0]
    pitch = model_orientation[1]
    yaw = model_orientation[2]

    quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
    model_pose.orientation.x = quaternion[0]
    model_pose.orientation.y = quaternion[1]
    model_pose.orientation.z = quaternion[2]
    model_pose.orientation.w = quaternion[3]

    return model_pose


def get_camera_pose_in_grasp_frame():
    camera_pose = Pose()

    #this will back the camera off along the approach direction 2 meters
    camera_pose.position.z -= 2

    #the camera points along the x direction, and we need it to point along the z direction
    roll = -math.pi/2.0
    pitch = -math.pi/2.0
    yaw = 0
    quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)

    camera_pose.orientation.x = quaternion[0]
    camera_pose.orientation.y = quaternion[1]
    camera_pose.orientation.z = quaternion[2]
    camera_pose.orientation.w = quaternion[3]

    return camera_pose


if __name__ == '__main__':
    saveImages = False

    t = time.localtime()
    minute = str(t.tm_min)
    if len(minute) == 1:
        minute = '0' + minute
    t_string = str(t.tm_mon) + "_" + str(t.tm_mday) + "_" + str(t.tm_hour) + "_" + minute

    output_image_dir = os.path.expanduser(GDL_DATA_PATH + "/rgbd_images_%s/" % t_string)
    models_dir = GRASPABLE_MODEL_PATH

    kinect_manager = GazeboKinectManager()
    kinect_manager.spawn_kinect()

    pause_physics_service_proxy = rospy.ServiceProxy("/gazebo/pause_physics", std_srvs.srv.Empty)
    unpause_physics_service_proxy = rospy.ServiceProxy("/gazebo/unpause_physics", std_srvs.srv.Empty)

    pause_physics_service_proxy()

    model_manager = GazeboModelManager(models_dir=models_dir)

    for model_name in os.listdir(GDL_GRASPS_PATH):


        model_output_image_dir = output_image_dir + model_name + '/'
        if not os.path.exists(model_output_image_dir):
            os.makedirs(model_output_image_dir)

        model_type = model_name
        model_manager.spawn_model(model_name, model_type)

        transform_manager = TransformerManager()

        camera_pose_in_world_frame = model_manager.get_model_state(kinect_manager.camera_name).pose
        transform_manager.add_transform(camera_pose_in_world_frame, "World", "Camera")

        model_pose_in_world_frame = model_manager.get_model_state(model_name).pose
        transform_manager.add_transform(model_pose_in_world_frame, "World", "Model")

        grasps = get_model_grasps(model_name)

        dataset = h5py.File(model_output_image_dir + "rgbd_and_labels.h5")
        print "Dataset is at: %s" % (model_output_image_dir + "rgbd_and_labels.h5")
        num_images = len(grasps)

        dataset.create_dataset("rgbd", (num_images, 480, 640, 4), chunks=(10, 480, 640, 4))
        dataset.create_dataset("labels", (num_images, 480, 640), chunks=(10, 480, 640))
        dataset.create_dataset("rgbd_patches", (num_images, 72, 72, 4), chunks=(10, 72, 72, 4))
        dataset.create_dataset("rgbd_patch_labels", (num_images, 1))

        for index in range(len(grasps)):
            print "%s / %s" % (index, len(grasps))
            grasp = grasps[index]

            transform_manager.add_transform(grasp.pose, "Model", "Grasp")
            sleep(0.5)

            camera_pose_in_grasp_frame = get_camera_pose_in_grasp_frame()

            camera_pose_in_world_frame = transform_manager.transform_pose(camera_pose_in_grasp_frame, "Grasp", "World")

            kinect_manager.set_model_state(camera_pose_in_world_frame.pose)

            transform_manager.add_transform(camera_pose_in_world_frame.pose, "World", "Camera")

            grasp_in_camera_frame = transform_manager.transform_pose(grasp.pose, "Grasp", "Camera").pose

            sleep(0.5)

            rgbd_image = np.copy(kinect_manager.get_rgbd_image())

            grasp_points = np.zeros((480, 640))
            overlay = np.copy(rgbd_image[:, :, 0])
            overlay_color = np.copy(rgbd_image[:, :, 0:3])

            #this is the pixel location of the grasp point
            u, v = xyz_to_uv((grasp_in_camera_frame.position.x,
                              grasp_in_camera_frame.position.y,
                              grasp_in_camera_frame.position.z))

            if u < overlay.shape[0]-2 and u > -2 and v < overlay.shape[1]-2 and v > -2:
                overlay[u-2:u+2, v-2:v+2] = grasp.energy
                overlay_color[u-2:u+2, v-2:v+2] = [0, 0, 255]
                overlay_color[238:242, 318:322] = [255, 0, 0]
                # u, v don't work properly so for now, setting grasp points manually
                grasp_points[240, 320] = grasp.energy

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

            #normalize overlay_color:
            overlay_color[:, :, 0:3] = overlay_color[:, :, 0:3]/255.0

            #normalize grasp_points
            #all nonzero grasp points are currently negative, so divide by the min.
            grasp_points = grasp_points/grasp_points.min()

            # u,v don't work properly
            #dataset["rgbd_patches"][index] = np.copy(rgbd_image[u-36:u+36, v-36:v+36, :])

            dataset["rgbd"][index] = np.copy(rgbd_image)
            dataset["grasp_energy"][index] = grasp.energy
            dataset["joint_angles"][index] = grasp.joint_angles
            dataset["joint_values"][index] = grasp.joint_values

            if saveImages:
                misc.imsave(output_filepath + "/" + 'out.png', grasp_points)
                misc.imsave(output_filepath + "/" + 'overlay.png', overlay)
                misc.imsave(output_filepath + "/" + 'overlay_color.png', overlay_color)
                misc.imsave(output_filepath + "/" + 'rgb.png', rgbd_image[:, :, 0:3])
                misc.imsave(output_filepath + "/" + 'r.png', rgbd_image[:, :, 0])
                misc.imsave(output_filepath + "/" + 'g.png', rgbd_image[:, :, 1])
                misc.imsave(output_filepath + "/" + 'b.png', rgbd_image[:, :, 2])
                misc.imsave(output_filepath + "/" + 'd.png', rgbd_image[:, :, 3])

        model_manager.remove_model(model_name)
