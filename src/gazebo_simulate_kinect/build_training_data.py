#!/usr/bin/env python
import rospy
import rospkg

from geometry_msgs.msg import Pose

import numpy as np
import os
from time import sleep
import time
import math
import tf
import random
from scipy import misc
import h5py
from gazebo_ros import gazebo_interface

from src.gazebo_model_manager import GazeboKinectManager, GazeboModelManager
from src.grasp import get_model_grasps
from src.transformer_manager import TransformerManager
from src.xyz_to_pixel_loc import xyz_to_uv
import copy

rospack = rospkg.RosPack()

GDL_DATA_PATH = os.environ["GDL_PATH"] + "/data"
#GRASPABLE_MODEL_PATH = os.environ["GDL_OBJECT_PATH"]
GDL_GRASPS_PATH = os.environ["GDL_GRASPS_PATH"]
GDL_MODEL_PATH = os.environ["GDL_MODEL_PATH"] + "/big_bird_models_processed"

NUM_VIRTUAL_CONTACTS = 16

#the +1 is for the center of the palm.
NUM_RGBD_PATCHES_PER_IMAGE = NUM_VIRTUAL_CONTACTS + 1
NUM_DOF = 4


def build_camera_pose_in_grasp_frame(grasp):
    camera_pose = Pose()

    #this will back the camera off along the approach direction .5 meters
    camera_pose.position.z -= .5

    #the camera points along the x direction, and we need it to point along the z direction
    roll = -math.pi/2.0
    pitch = -math.pi/2.0

    # rotate so that camera is upright.
    yaw = -math.pi/2.0 + grasp.joint_angles[1]

    quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)

    camera_pose.orientation.x = quaternion[0]
    camera_pose.orientation.y = quaternion[1]
    camera_pose.orientation.z = quaternion[2]
    camera_pose.orientation.w = quaternion[3]

    return camera_pose



def calculate_palm_and_vc_image_locations(grasp_in_camera_frame, transform_manager, grasp, graspNum):
    vc_uvds = []

    #this is the pixel location of the grasp point
    u, v, d = xyz_to_uv((grasp_in_camera_frame.position.x, grasp_in_camera_frame.position.y, grasp_in_camera_frame.position.z))

    vc_uvds.append((u, v, d))

    for i in range(len(grasp.virtual_contacts)):
        pose = Pose()
        pose.position.x = grasp.virtual_contacts[i][0]
        pose.position.y = grasp.virtual_contacts[i][1]
        pose.position.z = grasp.virtual_contacts[i][2]

        pose_in_world_frame = transform_manager.transform_pose(pose, "Model", "World").pose
        pose_in_camera_frame = transform_manager.transform_pose(pose, "Model", "Camera").pose

        u, v, d= xyz_to_uv((pose_in_camera_frame.position.x, pose_in_camera_frame.position.y, pose_in_camera_frame.position.z))
        #model_manager.spawn_sphere("sphere-%s-%s" % (graspNum, i),
        #                           pose_in_world_frame.position.x,
        #                           pose_in_world_frame.position.y,
        #                           pose_in_world_frame.position.z)
        vc_uvs.append((u, v, d))

    #sleep(1)

    return vc_uvds


def fill_images_with_grasp_points(vc_uvds, grasp, rgbd_image):

    grasp_points = np.zeros((len(vc_uvds), 480, 640))
    overlay = np.copy(rgbd_image[:, :, 0])
    rgbd_patches = np.zeros((len(vc_uvds), 72, 72, 4))

    for i in range(len(vc_uvds)):
        vc_u, vc_v, vc_d = vc_uvds[i]
        try:
            overlay[vc_u-2:vc_u+2, vc_v-2:vc_v+2] = 1.0
            grasp_points[i, vc_u, vc_v] = grasp.energy
            rgbd_patches[i] = rgbd_image[vc_u-36:vc_u+36, vc_v-36:vc_v+36, :]
        except Exception as e:
            print "u,v probably outside of image"
            print e

    return rgbd_patches, grasp_points, overlay


def create_save_path(model_output_image_dir, model_name, index ):
    output_filepath = model_output_image_dir + model_name + "_" + str(index)
    if not os.path.exists(output_filepath):
        os.makedirs(output_filepath)

    if not os.path.exists(model_output_image_dir + "overlays"):
        os.makedirs(model_output_image_dir + "overlays")

    return output_filepath


def update_transforms(transform_manager, grasp, kinect_manager):
    transform_manager.add_transform(grasp.pose, "Model", "Grasp")

    camera_pose_in_grasp_frame = build_camera_pose_in_grasp_frame(grasp)

    camera_pose_in_world_frame = transform_manager.transform_pose(camera_pose_in_grasp_frame, "Grasp", "World")

    #return false if the camera is below the XY plane (ie below the object)
    if camera_pose_in_world_frame.pose.position.z < 0:
        return False

    kinect_manager.set_model_state(camera_pose_in_world_frame.pose)

    transform_manager.add_transform(camera_pose_in_world_frame.pose, "World", "Camera")

    return True


def get_date_string():
    t = time.localtime()
    minute = str(t.tm_min)
    if len(minute) == 1:
        minute = '0' + minute
    t_string = str(t.tm_mon) + "_" + str(t.tm_mday) + "_" + str(t.tm_hour) + "_" + minute
    return t_string


if __name__ == '__main__':
    saveImages = False

    output_image_dir = os.path.expanduser(GDL_DATA_PATH + "/rgbd_images/%s/" % get_date_string())
    sleep(2)
    models_dir = GDL_MODEL_PATH

    model_manager = GazeboModelManager(models_dir=models_dir)
    model_manager.pause_physics()
    model_manager.clear_world()

    sleep(0.5)

    kinect_manager = GazeboKinectManager()

    if kinect_manager.get_model_state().status_message == 'GetModelState: model does not exist':
        kinect_manager.spawn_kinect()

    sleep(0.5)


    model_names = os.listdir(GDL_MODEL_PATH)
    firstTime = True

    for model_name in os.listdir(GDL_GRASPS_PATH):

        grasps = get_model_grasps(model_name)

        #do we have grasps for this model
        if not grasps:
            print str(model_name) + ' has no grasps'
            continue

        #do we actually have the model
        if not model_name in model_names:
            print 'we have grasps for ' + str(model_name) + ' but the model is not in the models directory'
            continue

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

        dataset_fullfilename = model_output_image_dir + "rgbd_and_labels.h5"

        if os.path.isfile(dataset_fullfilename):
            os.remove(dataset_fullfilename)

        dataset = h5py.File(dataset_fullfilename)
        print "Dataset is at: %s" % (dataset_fullfilename)

        num_images = len(grasps)
        #if num_images > 7:
        #    num_images = 7

        chunk_size = 10
        if num_images < 10:
            chunk_size = num_images

        dataset.create_dataset("rgbd", (num_images, 480, 640, 4), chunks=(chunk_size, 480, 640, 4))
        dataset.create_dataset("labels", (num_images, NUM_RGBD_PATCHES_PER_IMAGE, 480, 640), chunks=(chunk_size, NUM_RGBD_PATCHES_PER_IMAGE, 480, 640))
        dataset.create_dataset("rgbd_patches", (num_images, NUM_RGBD_PATCHES_PER_IMAGE, 72, 72, 4), chunks=(chunk_size, NUM_RGBD_PATCHES_PER_IMAGE, 72, 72, 4))
        dataset.create_dataset("rgbd_patch_labels", (num_images, 1))
        dataset.create_dataset("dof_values", (num_images, NUM_DOF), chunks=(chunk_size, NUM_DOF))
        dataset.create_dataset("uvd", (num_images, NUM_RGBD_PATCHES_PER_IMAGE, 3), chunks=(chunk_size, NUM_RGBD_PATCHES_PER_IMAGE, 3))

        for index in range(num_images):
            if firstTime:
                import pdb; pdb.set_trace()
                firstTime = False

            print "%s / %s grasps for %s" % (index, num_images, model_name)
            grasp = grasps[index]

            if not update_transforms(transform_manager, grasp, kinect_manager):
                print "Camera below model... skipping this grasp"
                continue
                # go to next index if the camera is positioned below the object

            grasp_in_camera_frame = transform_manager.transform_pose(grasp.pose, "Model", "Camera").pose

            #vc_uvs is a list of (u,v) tuples in the camera_frame representing:
            #1)the palm,
            #2)all the virtual contacts used in graspit
            vc_uvds = calculate_palm_and_vc_image_locations(grasp_in_camera_frame, transform_manager, grasp, index)

            #this is a processed rgbd_image that has been normalized and any nans have been removed
            rgbd_image = np.copy(kinect_manager.get_normalized_rgbd_image())

            rgbd_patches, grasp_points, overlay = fill_images_with_grasp_points(vc_uvds, grasp, rgbd_image)

            output_filepath = create_save_path(model_output_image_dir, model_name, index)

            dataset["rgbd_patches"][index] = np.copy(rgbd_patches)
            dataset["rgbd_patch_labels"][index] = grasp.energy
            dataset["rgbd"][index] = np.copy(rgbd_image)
            dataset["labels"][index] = np.copy(grasp_points)
            dataset["dof_values"][index] = np.copy(grasp.dof_values[1:])
            dataset["uvd"][index] = vc_uvds

            misc.imsave(output_filepath + "/" + 'overlay.png', overlay)
            misc.imsave(model_output_image_dir + "overlays" + "/" + 'overlay' + str(index) + '.png', overlay)
            misc.imsave(output_filepath + "/" + 'rgb.png', rgbd_image[:, :, 0:3])
            misc.imsave(output_filepath + "/" + 'd.png', rgbd_image[:, :, 3])

            #for i in range(len(grasp.virtual_contacts)):
            #    model_manager.remove_model("sphere-%s-%s" % (index, i))
            #sleep(1)

        model_manager.remove_model(model_name)
