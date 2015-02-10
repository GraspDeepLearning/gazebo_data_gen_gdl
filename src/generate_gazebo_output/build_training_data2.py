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
from src.transformer_manager import TransformerManager
from src.xyz_to_pixel_loc import xyz_to_uv
import copy
from grasp_dataset import GraspDataset

rospack = rospkg.RosPack()

#GDL_DATA_PATH = os.environ["GDL_PATH"] + "/data"
GDL_DATA_PATH = "/media/Elements/gdl_data"

GDL_GRASPS_PATH = os.environ["GDL_GRASPS_PATH"]
GDL_MODEL_PATH = os.environ["GDL_MODEL_PATH"] + "/big_bird_models_processed"

NUM_VIRTUAL_CONTACTS = 16

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



def calculate_palm_and_vc_image_locations(grasp_in_camera_frame, transform_manager, grasp):
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

def update_transforms(transform_manager, grasp_pose, kinect_manager, cameraDist):
    transform_manager.add_transform(grasp_pose, "Model", "Grasp")

    camera_pose_in_grasp_frame = build_camera_pose_in_grasp_frame(grasp_pose, cameraDist)

    camera_pose_in_world_frame = transform_manager.transform_pose(camera_pose_in_grasp_frame, "Grasp", "World")
    grasp_pose_in_world_frame = transform_manager.transform_pose(grasp_pose, "Model", "World")

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

    graspit_grasps_dir = os.path.expanduser("~/grasp_deep_learning/data/grasp_datasets/contact_and_potential_grasps/")
    graspit_agg_dir = os.path.expanduser("~/grasp_deep_learning/data/grasp_datasets/")

    rospack = rospkg.RosPack()
    DATASET_TEMPLATE_PATH = rospack.get_path('grasp_dataset')
    graspit_grasp_dataset = GraspDataset(graspit_agg_dir + "contact_and_potential_grasps.h5",
                                 DATASET_TEMPLATE_PATH + "/dataset_configs/graspit_grasps_dataset.yaml")

    gazebo_grasp_dataset = GraspDataset(graspit_agg_dir + "gazebo_contact_and_potential_grasps" + get_date_string() + ".h5",
                                        DATASET_TEMPLATE_PATH + "/dataset_configs/gazebo_capture_config.yaml")
    numIter = 0
    for grasp in graspit_grasp_dataset.random_iterator(num_items=51):
        numIter += 1
        model_name = grasp.model_name[0]

        grasp_pose = Pose()
        grasp_pose.position.x = grasp.palm_pose[0]
        grasp_pose.position.y = grasp.palm_pose[1]
        grasp_pose.position.z = grasp.palm_pose[2]
        grasp_pose.orientation.x = grasp.palm_pose[3]
        grasp_pose.orientation.y = grasp.palm_pose[4]
        grasp_pose.orientation.z = grasp.palm_pose[5]
        grasp_pose.orientation.w = grasp.palm_pose[6]

        model_manager.spawn_model(model_name=model_name,  model_type=model_name)

        transform_manager = TransformerManager()

        camera_pose_in_world_frame = model_manager.get_model_state(kinect_manager.camera_name).pose
        transform_manager.add_transform(camera_pose_in_world_frame, "World", "Camera")

        model_pose_in_world_frame = model_manager.get_model_state(model_name).pose
        transform_manager.add_transform(model_pose_in_world_frame, "World", "Model")

        print "%s: Running %s..." % (numIter, model_name)

        #break so that we can manually add lights to gazebo
        # if firstTime:
        #     import pdb; pdb.set_trace()
        #     firstTime = False

        updated_transforms, wrist_roll = update_transforms(transform_manager, grasp_pose, kinect_manager, cameraDist)

        if not updated_transforms:
            print "Camera below model... skipping this grasp"
            # go to next index if the camera is positioned below the object
        else:
            grasp_in_camera_frame = transform_manager.transform_pose(grasp_pose, "Model", "Camera").pose

            #vc_uvs is a list of (u,v) tuples in the camera_frame representing:
            #1)the palm,
            #2)all the virtual contacts used in graspit
            vc_uvds = calculate_palm_and_vc_image_locations(grasp_in_camera_frame, transform_manager, grasp)

            #this is a processed rgbd_image that has been normalized and any nans have been removed
            rgbd_image = np.copy(kinect_manager.get_normalized_rgbd_image())

            # Sometimes for no real reason openni fails and the depths stop being published.
            if rgbd_image[:, :, 3].max() == 0.0:
                print "SOMETHING'S BROKEN!"
                import IPython; IPython.embed()

            #rgbd_patches, grasp_points, overlay = fill_images_with_grasp_points(vc_uvds, grasp, rgbd_image)

            gazebo_grasp = gazebo_grasp_dataset.Grasp(
                rgbd=rgbd_image,
                dof_values=grasp.dof_values,
                palm_pose=grasp.palm_pose,
                joint_values=grasp.joint_values,
                uvd=vc_uvds,
                wrist_roll=wrist_roll,
                virtual_contacts=grasp.virtual_contacts,
                model_name=grasp.model_name,
                energy=grasp.energy
            )

            gazebo_grasp_dataset.add_grasp(gazebo_grasp)

        model_manager.remove_model(model_name)
