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

#GDL_DATA_PATH = os.environ["GDL_PATH"] + "/data"
GDL_DATA_PATH = "/media/Elements/gdl_data"

GDL_GRASPS_PATH = os.environ["GDL_GRASPS_PATH"]
GDL_MODEL_PATH = os.environ["GDL_MODEL_PATH"] + "/big_bird_models_processed"

NUM_VIRTUAL_CONTACTS = 7

#the +1 is for the center of the palm.
NUM_RGBD_PATCHES_PER_IMAGE = NUM_VIRTUAL_CONTACTS + 1
NUM_DOF = 4
PATCH_SIZE = 170


def build_camera_pose_in_grasp_frame(grasp, cameraDist):
    camera_pose = Pose()

    #this will back the camera off along the approach direction 'cameraDist' meters
    camera_pose.position.z -= cameraDist

    # quaternion = (grasp.pose.orientation.x, grasp.pose.orientation.y,grasp.pose.orientation.z,grasp.pose.orientation.w)
    # quaternion_inverse = tf.transformations.quaternion_inverse(quaternion)

    # r,p,y = tf.transformations.euler_from_quaternion(quaternion_inverse)

    #the camera points along the x direction, and we need it to point along the z direction
    roll = 0
    pitch = -math.pi/2.0

    # rotate so that camera is upright.
    yaw = 0#-math.pi/2.0 + grasp.joint_angles[0]

    # #rotation matrix from grasp to model rather than model to grasp.
    # quaternion_inverse = tf.transformations.quaternion_inverse(*grasp.pose.orientation)
    # q= quaternion_inverse

    # yaw = math.atan2(2.0*(q.y*q.z + q.w*q.x), q.w*q.w - q.x*q.x - q.y*q.y + q.z*q.z)
    # #for intrinsic rotation
    # alpha, _, _ = tf.transformations.euler_from_quaternion(*quaternion_inverse,'szyx')

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

        u, v, d = xyz_to_uv((pose_in_camera_frame.position.x, pose_in_camera_frame.position.y, pose_in_camera_frame.position.z))
        #model_manager.spawn_sphere("sphere-%s-%s" % (graspNum, i),
        #                          pose_in_world_frame.position.x,
        #                          pose_in_world_frame.position.y,
        #                          pose_in_world_frame.position.z)
        vc_uvds.append((u, v, d))

    #sleep(1)

    return vc_uvds


def fill_images_with_grasp_points(vc_uvds, grasp, rgbd_image):

    grasp_points = np.zeros((len(vc_uvds), 480, 640))
    overlay = np.copy(rgbd_image[:, :, 0])
    rgbd_patches = np.zeros((len(vc_uvds), PATCH_SIZE, PATCH_SIZE, 4))

    for i in range(len(vc_uvds)):
        vc_u, vc_v, vc_d = vc_uvds[i]
        try:
            overlay[vc_u-2:vc_u+2, vc_v-2:vc_v+2] = 1.0
            grasp_points[i, vc_u, vc_v] = grasp.energy
            rgbd_patches[i] = rgbd_image[vc_u-PATCH_SIZE/2:vc_u+PATCH_SIZE/2, vc_v-PATCH_SIZE/2:vc_v+PATCH_SIZE/2, :]
        except Exception as e:
            print "u,v probably outside of image"
            print e

    return rgbd_patches, grasp_points, overlay


def create_save_path(model_output_image_dir, model_name, index ):
    #output_filepath = model_output_image_dir + model_name + "_" + str(index)
    #if not os.path.exists(output_filepath):
    #    os.makedirs(output_filepath)

    if not os.path.exists(model_output_image_dir + "overlays"):
        os.makedirs(model_output_image_dir + "overlays")

def update_transforms(transform_manager, grasp, kinect_manager, cameraDist):
    transform_manager.add_transform(grasp.pose, "Model", "Grasp")

    camera_pose_in_grasp_frame = build_camera_pose_in_grasp_frame(grasp, cameraDist)

    camera_pose_in_world_frame = transform_manager.transform_pose(camera_pose_in_grasp_frame, "Grasp", "World")
    grasp_pose_in_world_frame = transform_manager.transform_pose(grasp.pose, "Model", "World")

    #look at these as two points in world coords, ignoring rotations
    dx = camera_pose_in_world_frame.pose.position.x - grasp_pose_in_world_frame.pose.position.x
    dy = camera_pose_in_world_frame.pose.position.y - grasp_pose_in_world_frame.pose.position.y
    dz = camera_pose_in_world_frame.pose.position.z - grasp_pose_in_world_frame.pose.position.z

    #first find angle around world z to orient camera towards object
    rot = math.atan2(dy,dx)

    #now find angle to tilt camera down towards object
    dist_in_xy_plane = math.hypot(dx,dy)
    tilt = math.atan2(dz, dist_in_xy_plane)

    #now find rpy to rotate camera from 0,0,0,0 to rot, tilt

    roll = 0
    #make sure the camera is tilted up or down to center the palm vertically
    pitch = tilt
    #this centers the object in the x,y world plane. by rotating around world's z axis
    yaw = rot + math.pi

    quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)

    quat_grasp = camera_pose_in_world_frame.pose.orientation

    grasp_rpy =tf.transformations.euler_from_quaternion((quat_grasp.x, quat_grasp.y, quat_grasp.z, quat_grasp.w))
    camera_rpy = roll, pitch, yaw


    wrist_roll = grasp_rpy[0] - camera_rpy[0]

    camera_pose_in_world_frame.pose.orientation.x = quaternion[0]
    camera_pose_in_world_frame.pose.orientation.y = quaternion[1]
    camera_pose_in_world_frame.pose.orientation.z = quaternion[2]
    camera_pose_in_world_frame.pose.orientation.w = quaternion[3]


    #return false if the camera is below the XY plane (ie below the object)
    if camera_pose_in_world_frame.pose.position.z < 0:
        return False, False

    kinect_manager.set_model_state(camera_pose_in_world_frame.pose)

    transform_manager.add_transform(camera_pose_in_world_frame.pose, "World", "Camera")

    return True, wrist_roll


def get_date_string():
    t = time.localtime()
    minute = str(t.tm_min)
    if len(minute) == 1:
        minute = '0' + minute
    t_string = str(t.tm_mon) + "_" + str(t.tm_mday) + "_" + str(t.tm_hour) + "_" + minute
    return t_string


if __name__ == '__main__':
    saveImages = False
    cameraDist = 2.0

    output_image_dir = os.path.expanduser(GDL_DATA_PATH + "/rgbd_images/%sm-%s/" % (str(cameraDist), get_date_string()))
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

        dataset_fullfilename = model_output_image_dir + "rgbd_and_labels_temp.h5"

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

        dataset.create_dataset("rgbd_temp", (num_images, 480, 640, 4), chunks=(chunk_size, 480, 640, 4))
        dataset.create_dataset("rgbd_patches_temp", (num_images, NUM_RGBD_PATCHES_PER_IMAGE, PATCH_SIZE, PATCH_SIZE, 4), chunks=(chunk_size, NUM_RGBD_PATCHES_PER_IMAGE, PATCH_SIZE, PATCH_SIZE, 4))
        dataset.create_dataset("rgbd_patch_labels_temp", (num_images, 1))
        dataset.create_dataset("dof_values_temp", (num_images, NUM_DOF), chunks=(chunk_size, NUM_DOF))
        dataset.create_dataset("palm_pose_temp", (num_images, 7  ), chunks=(chunk_size, 7))
        dataset.create_dataset("joint_values_temp", (num_images, 8  ), chunks=(chunk_size, 8))
        dataset.create_dataset("uvd_temp", (num_images, NUM_RGBD_PATCHES_PER_IMAGE, 3), chunks=(chunk_size, NUM_RGBD_PATCHES_PER_IMAGE, 3))
        dataset.create_dataset("wrist_roll", (num_images, 1))
        dataset_index = 0

        print "Running %s..." % model_name

        for index in range(num_images):
            if firstTime:
                import pdb; pdb.set_trace()
                firstTime = False

            print "%s / %s grasps for %s" % (index, num_images, model_name)
            grasp = grasps[index]

            updated_transforms, wrist_roll = update_transforms(transform_manager, grasp, kinect_manager, cameraDist)
            if not updated_transforms:
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

            # Sometimes for no real reason openni fails and the depths stop being published.
            if rgbd_image[:, :, 3].max() == 0.0:
                print "SOMETHING'S BROKEN!"
                import IPython; IPython.embed()

            rgbd_patches, grasp_points, overlay = fill_images_with_grasp_points(vc_uvds, grasp, rgbd_image)

            grasp_pose_array = [grasp.pose.position.x,
                            grasp.pose.position.y, 
                            grasp.pose.position.z,
                            grasp.pose.orientation.x,
                            grasp.pose.orientation.y,
                            grasp.pose.orientation.z,
                            grasp.pose.orientation.w]

            
            if dataset_index % 100 == 0 or saveImages:
                create_save_path(model_output_image_dir, model_name, index)
                misc.imsave(model_output_image_dir + "overlays" + "/" + 'overlay' + str(index) + '.png', overlay)
                misc.imsave(model_output_image_dir + "overlays" + "/" + 'depth' + str(index) + '.png', rgbd_image[:, :, 3])

            dataset["rgbd_patches_temp"][dataset_index] = np.copy(rgbd_patches)
            dataset["rgbd_patch_labels_temp"][dataset_index] = grasp.energy

            # Sometimes for no real reason openni fails and the depths stop being published.
            if rgbd_image[:, :, 3].max() == 0.0:
                print "SOMETHING'S BROKEN!"
                import IPython; IPython.embed()

            dataset["rgbd_temp"][dataset_index] = np.copy(rgbd_image)
            dataset["dof_values_temp"][dataset_index] = np.copy(grasp.dof_values)
            dataset["palm_pose_temp"][dataset_index] = np.copy(grasp_pose_array)
            dataset["joint_values_temp"][dataset_index] = np.copy(grasp.joint_angles)
            dataset["uvd_temp"][dataset_index] = vc_uvds
            dataset['wrist_roll'][dataset_index] = wrist_roll
            dataset_index += 1

            #for i in range(len(grasp.virtual_contacts)):
            #   model_manager.remove_model("sphere-%s-%s" % (index, i))
            #sleep(1)

        dataset_size = dataset_index
        chunk_size = 10    

        if dataset_size < 10:
            chunk_size = dataset_size

        dataset_fullfilename = model_output_image_dir + "rgbd_and_labels.h5"
        final_dataset = h5py.File(dataset_fullfilename)

        final_dataset.create_dataset("rgbd", (dataset_size, 480, 640, 4), chunks=(chunk_size, 480, 640, 4))
        final_dataset.create_dataset("rgbd_patches", (dataset_size, NUM_RGBD_PATCHES_PER_IMAGE, PATCH_SIZE, PATCH_SIZE, 4), chunks=(chunk_size, NUM_RGBD_PATCHES_PER_IMAGE, PATCH_SIZE, PATCH_SIZE, 4))
        final_dataset.create_dataset("rgbd_patch_labels", (dataset_size, 1))
        final_dataset.create_dataset("dof_values", (dataset_size, NUM_DOF), chunks=(chunk_size, NUM_DOF))
        final_dataset.create_dataset("palm_pose", (dataset_size, 7  ), chunks=(chunk_size, 7))
        final_dataset.create_dataset("joint_values", (dataset_size, 8  ), chunks=(chunk_size, 8))
        final_dataset.create_dataset("uvd", (dataset_size, NUM_RGBD_PATCHES_PER_IMAGE, 3), chunks=(chunk_size, NUM_RGBD_PATCHES_PER_IMAGE, 3))
        final_dataset.create_dataset("wrist_roll", (dataset_size, 1))

        for i in range(dataset_size):
            final_dataset["rgbd"][i] = dataset["rgbd_temp"][i]
            final_dataset["rgbd_patches"][i] = dataset["rgbd_patches_temp"][i]
            final_dataset["rgbd_patch_labels"][i] = dataset["rgbd_patch_labels_temp"][i]
            final_dataset["dof_values"][i] = dataset["dof_values_temp"][i]
            final_dataset["palm_pose"][i] = dataset["palm_pose_temp"][i]
            final_dataset["joint_values"][i] = dataset["joint_values_temp"][i]
            final_dataset["uvd"][i] = dataset["uvd_temp"][i]
            final_dataset["wrist_roll"][i] = dataset["wrist_roll"][i]

        dataset.close()
        final_dataset.close()
        model_manager.remove_model(model_name)
