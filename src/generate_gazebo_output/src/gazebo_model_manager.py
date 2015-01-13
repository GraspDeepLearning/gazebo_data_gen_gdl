#!/usr/bin/env python
import rospy
import rospkg
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from gazebo_ros import gazebo_interface
from gazebo_msgs.srv import (DeleteModelRequest, DeleteModelResponse, DeleteModel, 
                             GetModelState, GetModelStateRequest, GetWorldProperties,
                             SetModelState, SetModelStateRequest)

import numpy as np
import os
from time import sleep

import tf_conversions
import PyKDL
import math
import std_srvs.srv

rospack = rospkg.RosPack()


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

        self.get_model_state_service = rospy.ServiceProxy(gazebo_namespace + '/get_model_state', GetModelState)
        self.set_model_state_service = rospy.ServiceProxy(gazebo_namespace + '/set_model_state', SetModelState)

    def spawn_kinect(self):
        model_xml = rospy.get_param("robot_description")

        #f = PyKDL.Frame(PyKDL.Rotation.RPY(0, math.pi, math.pi), PyKDL.Vector(0, 0, 2))
        f = PyKDL.Frame(PyKDL.Rotation.RPY(0, math.pi+math.pi/4.0, math.pi), PyKDL.Vector(0, 0, 2))
        f = PyKDL.Frame(PyKDL.Rotation.RPY(0, math.pi/4.0, 0), PyKDL.Vector(0, 0, 2))
        model_pose = tf_conversions.posemath.toMsg(f)
        #model_pose = Pose()
        robot_namespace = self.camera_name
        gazebo_interface.spawn_urdf_model_client(model_name=self.camera_name,
                                                model_xml=model_xml,
                                                robot_namespace=robot_namespace,
                                                initial_pose=model_pose,
                                                reference_frame="world",
                                                gazebo_namespace=self.gazebo_namespace)

    def get_normalized_rgbd_image(self):
        rgbd_image = np.copy(self.rgbd_listener.rgbd_image)

        #fix nans in depth
        max_depth = np.nan_to_num(rgbd_image[:, :, 3]).max()*1.3 + 5.0
        for x in range(rgbd_image.shape[0]):
            for y in range(rgbd_image.shape[1]):
                if rgbd_image[x, y, 3] != rgbd_image[x, y, 3]:
                    rgbd_image[x, y, 3] = max_depth

        return rgbd_image

    def get_model_state(self):
        get_model_state_req = GetModelStateRequest()
        get_model_state_req.model_name = self.camera_name
        return self.get_model_state_service(get_model_state_req)

    def set_model_state(self, pose=Pose()):
        set_model_state_req = SetModelStateRequest()
        set_model_state_req.model_state.model_name = self.camera_name
        set_model_state_req.model_state.pose = pose
        return self.set_model_state_service(set_model_state_req)



class GazeboModelManager():

    def __init__(self,
                 models_dir,
                 gazebo_namespace="/gazebo"):

        self.gazebo_namespace = gazebo_namespace
        self.models_dir = models_dir
        self.delete_model_service_proxy = rospy.ServiceProxy(gazebo_namespace + '/delete_model', DeleteModel)
        self.get_model_state_service_proxy = rospy.ServiceProxy(gazebo_namespace + '/get_model_state', GetModelState)
        self.set_model_state_service_proxy = rospy.ServiceProxy(gazebo_namespace + '/set_model_state', SetModelState)
        self.get_world_properties_proxy = rospy.ServiceProxy(gazebo_namespace + '/get_world_properties', GetWorldProperties)
        self.pause_physics_service_proxy = rospy.ServiceProxy(gazebo_namespace + "/pause_physics", std_srvs.srv.Empty)
        self.unpause_physics_service_proxy = rospy.ServiceProxy(gazebo_namespace + "/unpause_physics", std_srvs.srv.Empty)

    def remove_model(self, model_name="coke_can"):

        del_model_req = DeleteModelRequest(model_name)
        self.delete_model_service_proxy(del_model_req)

    def clear_world(self):
        world_properties = self.get_world_properties_proxy()
        for model_name in world_properties.model_names:
            if model_name != "camera1":
                self.remove_model(model_name)

    def pause_physics(self):
        self.pause_physics_service_proxy()

    def unpause_physics(self):
        self.unpause_physics_service_proxy()

    def spawn_model(self, model_name="coke_can", model_type="coke_can", model_pose=None):
        model_xml = open(self.models_dir + "/" + model_type + "/model.sdf").read()
        if not model_pose:
            model_pose = Pose()
            model_pose.position.x = 0
            model_pose.position.y = 0
            model_pose.position.z = 0
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

    def spawn_sphere(self, sphere_name="0", x=0, y=0, z=0):

        model_xml = open(os.environ["GDL_PATH"] + "/models/sphere/model.sdf").read()
        model_pose = Pose()
        model_pose.position.x = x
        model_pose.position.y = y
        model_pose.position.z = z
        robot_namespace = sphere_name
        gazebo_interface.spawn_sdf_model_client(model_name=sphere_name,
                                                model_xml=model_xml,
                                                robot_namespace=robot_namespace,
                                                initial_pose=model_pose,
                                                reference_frame="world",
                                                gazebo_namespace=self.gazebo_namespace)


    def does_world_contain_model(self, model_name="coke_can"):
        get_model_state_req = GetModelStateRequest()
        get_model_state_req.model_name = model_name
        resp = self.get_model_state_service_proxy(get_model_state_req)
        return resp.success

    def get_model_state(self, model_name="coke_can"):
        get_model_state_req = GetModelStateRequest()
        get_model_state_req.model_name = model_name
        return self.get_model_state_service_proxy(get_model_state_req)

    def set_model_state(self, model_name="coke_can", pose=Pose()):
        set_model_state_req = SetModelStateRequest()
        set_model_state_req.model_state.model_name = model_name
        set_model_state_req.model_state.pose = pose
        return self.set_model_state_service_proxy(set_model_state_req)
