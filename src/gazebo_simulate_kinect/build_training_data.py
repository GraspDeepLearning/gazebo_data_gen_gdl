#!/usr/bin/env python
import rospy
import rospkg
rospack = rospkg.RosPack()
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from gazebo_ros import gazebo_interface
from gazebo_msgs.srv import DeleteModelRequest, DeleteModel, DeleteModelResponse, GetModelState, GetModelStateRequest, SetModelState, SetModelStateRequest

import numpy as np
import os
from time import sleep
import matplotlib.pyplot as plt

import tf_conversions
import xyz_to_pixel_loc
import PyKDL
import math
import std_srvs.srv

import tf
import random

from scipy import misc

import h5py

GAZEBO_MODEL_PATH = os.environ["GAZEBO_MODEL_PATH"]
GRASPABLE_MODEL_PATH = GAZEBO_MODEL_PATH
#GRASPABLE_MODEL_PATH = rospack.get_path('object_models') + "/models/cgdb/model_database/"


class RGBDListener():

    def __init__(self,
                 depth_topic="/camera1/camera/depth/image_raw",
                 rgb_topic="/camera1/camera/rgb/image_raw"):

        self.depth_topic = depth_topic
        self.rgb_topic = rgb_topic
        self.rgbd_image = np.zeros((480, 640, 4))

    def depth_image_callback(self, data):
        depth_image_np = self.image2numpy(data)
        self.rgbd_image[:, :, 3] = depth_image_np

    def rgb_image_callback(self, data):
        rgbd_image_np = self.image2numpy(data)
        self.rgbd_image[:, :, 0:3] = rgbd_image_np

    #this method from:
    #https://github.com/rll/sushichallenge/blob/master/python/brett2/ros_utils.py
    def image2numpy(self, image):
        if image.encoding == 'rgb8':
            return np.fromstring(image.data, dtype=np.uint8).reshape(image.height, image.width, 3)[:, :, ::-1]
        if image.encoding == 'bgr8':
            return np.fromstring(image.data, dtype=np.uint8).reshape(image.height, image.width, 3)
        elif image.encoding == 'mono8':
            return np.fromstring(image.data, dtype=np.uint8).reshape(image.height, image.width)
        elif image.encoding == '32FC1':
            return np.fromstring(image.data, dtype=np.float32).reshape(image.height, image.width)
        else:
            raise Exception

    def listen(self):

        rospy.init_node('listener', anonymous=True)

        rospy.Subscriber(self.depth_topic, Image, self.depth_image_callback, queue_size=1)
        rospy.Subscriber(self.rgb_topic, Image, self.rgb_image_callback, queue_size=1)


class GazeboKinectManager():
    def __init__(self, gazebo_namespace="/gazebo"):
        self.gazebo_namespace = gazebo_namespace
        self.rgbd_listener = RGBDListener()
        self.rgbd_listener.listen()
        self.camera_name = "camera1"

    def spawn_kinect(self):
        model_xml = rospy.get_param("robot_description")

        #f = PyKDL.Frame(PyKDL.Rotation.RPY(0, math.pi, math.pi), PyKDL.Vector(0, 0, 2))
        f = PyKDL.Frame(PyKDL.Rotation.RPY(0, math.pi+math.pi/4.0, math.pi), PyKDL.Vector(0, 0, 2))
        f = PyKDL.Frame(PyKDL.Rotation.RPY(0, math.pi/4.0, 0), PyKDL.Vector(0, 0, 2))
        model_pose = tf_conversions.posemath.toMsg(f)
        robot_namespace = self.camera_name
        gazebo_interface.spawn_urdf_model_client(model_name=self.camera_name,
                                                model_xml=model_xml,
                                                robot_namespace=robot_namespace,
                                                initial_pose=model_pose,
                                                reference_frame="world",
                                                gazebo_namespace=self.gazebo_namespace)

    def get_rgbd_image(self):
        return self.rgbd_listener.rgbd_image


class GazeboModelManager():

    def __init__(self,
                 gazebo_namespace="/gazebo",
                 models_dir=GAZEBO_MODEL_PATH):

        self.gazebo_namespace = gazebo_namespace
        self.models_dir = models_dir
        self.delete_model_service = rospy.ServiceProxy(gazebo_namespace + '/delete_model', DeleteModel)
        self.get_model_state_service = rospy.ServiceProxy(gazebo_namespace + '/get_model_state', GetModelState)
        self.set_model_state_service = rospy.ServiceProxy(gazebo_namespace + '/set_model_state', SetModelState)

    def remove_model(self, model_name="coke_can"):

        del_model_req = DeleteModelRequest(model_name)
        self.delete_model_service(del_model_req)

    def spawn_model(self, model_name="coke_can", model_pose=None):

        model_xml = open(self.models_dir + "/" + model_name + "/model.sdf").read()
        if not model_pose:
            model_pose = Pose()
            model_pose.position.x = 2
            model_pose.position.y = 0
            model_pose.position.z = 1
        robot_namespace = model_name
        gazebo_interface.spawn_sdf_model_client(model_name=model_name,
                                                model_xml=model_xml,
                                                robot_namespace=robot_namespace,
                                                initial_pose=model_pose,
                                                reference_frame="world",
                                                gazebo_namespace=self.gazebo_namespace)

        #large models can take a moment to load
        while not self.does_world_contain_model(model_name):
            sleep(0.5)

        sleep(2)

    def does_world_contain_model(self, model_name="coke_can"):
        get_model_state_req = GetModelStateRequest()
        get_model_state_req.model_name = model_name
        resp = self.get_model_state_service(get_model_state_req)
        return resp.success

    def get_model_state(self, model_name="coke_can"):
        get_model_state_req = GetModelStateRequest()
        get_model_state_req.model_name = model_name
        return self.get_model_state_service(get_model_state_req)

    def set_model_state(self, model_name="coke_can", pose=Pose()):
        set_model_state_req = SetModelStateRequest()
        set_model_state_req.model_state.model_name = model_name
        set_model_state_req.model_state.pose = pose
        return self.set_model_state_service(set_model_state_req)


def add_transform(pose_in_world_frame, frame_id, child_frame_id, transformer):
    transform_msg = tf.msg.geometry_msgs.msg.TransformStamped()
    transform_msg.transform.translation.x = pose_in_world_frame.position.x
    transform_msg.transform.translation.y = pose_in_world_frame.position.y
    transform_msg.transform.translation.z = pose_in_world_frame.position.z

    transform_msg.transform.rotation.x = pose_in_world_frame.orientation.x
    transform_msg.transform.rotation.y = pose_in_world_frame.orientation.y
    transform_msg.transform.rotation.z = pose_in_world_frame.orientation.z
    transform_msg.transform.rotation.w = pose_in_world_frame.orientation.w
    transform_msg.child_frame_id = child_frame_id
    transform_msg.header.frame_id = frame_id

    transformer.setTransform(transform_msg)


def transform_pose(pose, old_frame, new_frame, transformer):

    transform_msg = tf.msg.geometry_msgs.msg.PoseStamped()
    transform_msg.pose.position.x = pose.position.x
    transform_msg.pose.position.y = pose.position.y
    transform_msg.pose.position.z = pose.position.z

    transform_msg.pose.orientation.x = pose.orientation.x
    transform_msg.pose.orientation.y = pose.orientation.y
    transform_msg.pose.orientation.z = pose.orientation.z
    transform_msg.pose.orientation.w = pose.orientation.w

    transform_msg.header.frame_id = old_frame

    return transformer.transformPose(new_frame, transform_msg)


class Grasp():

    def __init__(self,energy, joint_angles, pose):
        self.energy = energy
        self.joint_angles = joint_angles
        self.pose = pose

def get_fake_model_grasps(model_name):

    return [Grasp()]

def get_model_grasps(model_name):

    grasps = []

    graspfilepath = rospack.get_path('training_grasps') + "/grasps/" + model_name
    for graspfile in  os.listdir(graspfilepath):
        new_grasps = graspfilepath_to_grasps(graspfilepath + "/" + graspfile)
        for grasp in new_grasps:
            grasps.append(grasp)

    return grasps


def graspfilepath_to_grasps(graspfilepath):

    grasps = []
    energy = 0
    joint_angles = []
    pose_array = []

    f = open(graspfilepath)
    for line in f.readlines():
        if "energy: " in line:
            energy = float(line[len("energy: "):])
        if "joint_angles: " in line:
            joint_angles = line[len("joint_angles: "):-1]
            #import IPython
            #IPython.embed()
            joint_angles = [float(joint_angle) for joint_angle in joint_angles.split()]
        if "pose: " in line:
            pose_array = line[len("pose: "):]
            pose_array = [float(x) for x in pose_array.split()][1:]
            pose = Pose()
            pose.position.x = pose_array[0]/1000.0
            pose.position.y = pose_array[1]/1000.0
            pose.position.z = pose_array[2]/1000.0
            pose.orientation.w = pose_array[3]
            pose.orientation.x = pose_array[4]
            pose.orientation.y = pose_array[5]
            pose.orientation.z = pose_array[6]
            grasps.append(Grasp(energy, joint_angles, pose))

    return grasps


def gen_model_pose(model_orientation):
    model_pose = Pose()
    model_pose.position.x = 2 + random.uniform(-.25,.25)
    model_pose.position.y = 0.0 + random.uniform(-.25,.25)
    model_pose.position.z = 1 + random.uniform(-.25,.25)

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

    output_image_dir = "rgbd_images/"
    models_dir = GRASPABLE_MODEL_PATH

    kinect_manager = GazeboKinectManager()
    kinect_manager.spawn_kinect()

    pause_physics_service_proxy = rospy.ServiceProxy("/gazebo/pause_physics", std_srvs.srv.Empty)
    unpause_physics_service_proxy = rospy.ServiceProxy("/gazebo/unpause_physics", std_srvs.srv.Empty)
    #we don't need any physics right now
    pause_physics_service_proxy()

    model_manager = GazeboModelManager(models_dir=models_dir)

    model_orientations = []
    for r in np.linspace(0, 2*math.pi, num=10):
        for p in np.linspace(0, 2*math.pi, num=10):
            for y in np.linspace(0, 2*math.pi, num=5):
                model_orientations.append((r, p, y))

    for model_name in ['coke_can']:

        model_manager.spawn_model(model_name)

        transformer = tf.TransformerROS(True, rospy.Duration(10.0))

        camera_pose_in_world_frame = model_manager.get_model_state(kinect_manager.camera_name).pose
        add_transform(camera_pose_in_world_frame, "World", "Camera", transformer)

        dataset = h5py.File(output_image_dir + "rgbd_and_labels.h5")
        num_images = len(model_orientations)

        dataset.create_dataset("rgbd", (num_images, 480, 640, 4), chunks=(10, 480, 640, 4))
        dataset.create_dataset("rgbd_labels", (num_images, 480, 640), chunks=(10, 480, 640))

        for index in range(len(model_orientations)):
            model_orientation = model_orientations[index]
            model_pose = gen_model_pose(model_orientation)

            model_manager.set_model_state(model_name, model_pose)

            sleep(1)

            rgbd_image = np.copy(kinect_manager.get_rgbd_image())

            grasp_points = np.zeros((480, 640))
            overlay = np.copy(rgbd_image[:, :, 0])

            model_pose_in_world_frame = model_manager.get_model_state(model_name).pose
            add_transform(model_pose_in_world_frame, "World", "Model", transformer)

            model_grasps = get_model_grasps(model_name)

            for model_grasp in model_grasps:

                #model_grasp.pose.position.x = 0
                #model_grasp.pose.position.y = 0
                #model_grasp.pose.position.z = 0

                add_transform(model_grasp.pose, "Model", "Grasp", transformer)

                #get grasp point in camera frame
                grasp_in_camera_frame = transform_pose(model_grasp.pose, "Grasp", "Camera", transformer)
                #grasp_in_camera_frame = transform_pose(model_grasp.pose, "Model", "Camera", transformer)
                grasp_in_camera_frame = grasp_in_camera_frame.pose

                #this is the pixel location of the grasp point
                u, v = xyz_to_pixel_loc.xyz_to_uv((grasp_in_camera_frame.position.x, grasp_in_camera_frame.position.y, grasp_in_camera_frame.position.z))

                #import IPython
                #IPython.embed()
                overlay[u-2:u+2, v-2:v+2] = model_grasp.energy
                grasp_points[u, v] = model_grasp.energy

            output_filepath = output_image_dir + model_name + "_" + str(index)
            if not os.path.exists(output_filepath):
                os.makedirs(output_filepath)

            #fix nans in depth
            rgbd_image[:, :, 3] = np.nan_to_num(rgbd_image[:, :, 3])

            #normalize rgb:
            rgbd_image[:, :, 0:3] = rgbd_image[:, :, 0:3]/255.0
            #normalize d
            rgbd_image[:, :, 3] = rgbd_image[:, :, 3]/rgbd_image[:, :, 3].max()
            #normalize grasp_points
            #all nonzero grasp points are currently negative, so divide by the min.
            grasp_points = grasp_points/grasp_points.min()

            dataset["rgbd"][index] = rgbd_image
            dataset["rgbd_labels"][index] = grasp_points

            misc.imsave(output_filepath + "/" + 'out.png', grasp_points)
            misc.imsave(output_filepath + "/" + 'overlay.png', overlay)
            misc.imsave(output_filepath + "/" + 'r.png', rgbd_image[:, :, 0])
            misc.imsave(output_filepath + "/" + 'g.png', rgbd_image[:, :, 1])
            misc.imsave(output_filepath + "/" + 'b.png', rgbd_image[:, :, 2])
            misc.imsave(output_filepath + "/" + 'd.png', rgbd_image[:, :, 3])

        model_manager.remove_model(model_name)
