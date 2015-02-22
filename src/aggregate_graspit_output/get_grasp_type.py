import tf
import math
import rospy

from time import sleep

from geometry_msgs.msg import Pose


class TransformerManager():
    def __init__(self):
        self.transformer = tf.TransformerROS(True, rospy.Duration(10.0))

    def add_transform(self, pose_in_world_frame, frame_id, child_frame_id):
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

        self.transformer.setTransform(transform_msg)
        sleep(.1)

    def transform_pose(self, pose, old_frame, new_frame):

        transform_msg = tf.msg.geometry_msgs.msg.PoseStamped()
        transform_msg.pose.position.x = pose.position.x
        transform_msg.pose.position.y = pose.position.y
        transform_msg.pose.position.z = pose.position.z

        transform_msg.pose.orientation.x = pose.orientation.x
        transform_msg.pose.orientation.y = pose.orientation.y
        transform_msg.pose.orientation.z = pose.orientation.z
        transform_msg.pose.orientation.w = pose.orientation.w

        transform_msg.header.frame_id = old_frame

        result = self.transformer.transformPose(new_frame, transform_msg)
        sleep(0.1)
        return result


def build_camera_pose_in_grasp_frame(camera_dist):
    camera_pose = Pose()

    #this will back the camera off along the approach direction 'cameraDist' meters
    camera_pose.position.z -= camera_dist

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


def get_wrist_roll(grasp_pose, transform_manager, camera_dist=2):

    transform_manager.add_transform(grasp_pose, "Model", "Grasp")

    camera_pose_in_grasp_frame = build_camera_pose_in_grasp_frame(camera_dist)
    camera_pose_in_model_frame = transform_manager.transform_pose(camera_pose_in_grasp_frame, "Grasp", "Model")

    #look at these as two points in world coords, ignoring rotations
    dx = camera_pose_in_model_frame.pose.position.x - grasp_pose.pose.position.x
    dy = camera_pose_in_model_frame.pose.position.y - grasp_pose.pose.position.y
    dz = camera_pose_in_model_frame.pose.position.z - grasp_pose.pose.position.z

    #first find angle around world z to orient camera towards object
    rot = math.atan2(dy, dx)

    #now find angle to tilt camera down towards object
    dist_in_xy_plane = math.hypot(dx, dy)
    tilt = math.atan2(dz, dist_in_xy_plane)

    #now find rpy to rotate camera from 0,0,0,0 to rot, tilt
    roll = 0
    #make sure the camera is tilted up or down to center the palm vertically
    pitch = tilt
    #this centers the object in the x,y world plane. by rotating around world's z axis
    yaw = rot + math.pi

    quat_grasp = camera_pose_in_model_frame.pose.orientation
    grasp_rpy = tf.transformations.euler_from_quaternion((quat_grasp.x, quat_grasp.y, quat_grasp.z, quat_grasp.w))
    camera_rpy = roll, pitch, yaw

    wrist_roll = grasp_rpy[0] - camera_rpy[0]

    return wrist_roll

