
import os

from collections import namedtuple
from geometry_msgs.msg import Pose

Grasp = namedtuple('Grasp', 'energy joint_angles dof_values pose virtual_contacts')


def get_model_grasps(model_name, graspClass=None):

    if graspClass is None:
        graspClass = Grasp

    grasps = []

    graspfilepath = os.path.expanduser(os.environ["GDL_GRASPS_PATH"] + "/" + model_name)

    if not os.path.exists(graspfilepath):
        print graspfilepath + " does not exist, skipping this object"
        return None

    for graspfile in os.listdir(graspfilepath):
        new_grasps = graspfilepath_to_grasps(graspfilepath + "/" + graspfile, graspClass)
        for grasp in new_grasps:
            grasps.append(grasp)

    return grasps


def graspfilepath_to_grasps(graspfilepath, graspClass):

    grasps = []

    f = open(graspfilepath)

    energy = 0
    joint_angles = []
    dof_values = []
    pose = Pose()
    virtual_contacts = []
    reading_vcs = False

    for line in f.readlines():
        if "energy: " in line:
            energy = float(line[len("energy: "):])
        #these are dof values
        elif "joint_angles: " in line:
            dof_values = line[len("joint_angles: "):-1]
            dof_values = [float(dof_value) for dof_value in dof_values.split()]
        if "jointValues: " in line:
            joint_angles = line[len("jointValues: "):-1]
            joint_angles = [float(joint_angle) for joint_angle in joint_angles.split()]
        elif "pose: " in line:
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
            # pose.orientation.w = pose_array[6]
            # pose.orientation.x = pose_array[3]
            # pose.orientation.y = pose_array[4]
            # pose.orientation.z = pose_array[5]

        elif "virtual contacts" in line:
            reading_vcs = True

        elif reading_vcs:
            vc_array = line.split()

            #check if we are finished
            if len(vc_array) != 3:
                grasps.append(graspClass(energy, joint_angles, dof_values[1:], pose, virtual_contacts))
                reading_vcs = False
                virtual_contacts = []

            else:
                x = float(vc_array[0])/1000.0
                y = float(vc_array[1])/1000.0
                z = float(vc_array[2])/1000.0

                virtual_contacts.append((x, y, z))




    return grasps