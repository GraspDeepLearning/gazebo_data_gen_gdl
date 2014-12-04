import numpy as np

k = np.array([589.366454, 0.0, 240.5, 0.0, 589.366454, 320.5, 0.0, 0.0, 1.0])
k = k.reshape((3, 3))


#point = np.array([x,y,z]) in camera frame
def xyz_to_uv(point_in_camera_frame):
    point_xz_flipped = np.array([point_in_camera_frame[2], point_in_camera_frame[1], point_in_camera_frame[0]])
    z = point_xz_flipped[2]
    point_norm = point_xz_flipped/(z*1.0)
    u, v, _ = np.dot(k, point_norm.T)

    v = 640 - v
    #u = 240 + 480 - u
    u = 480 - u

    #u and v are indices into the image
    return u, v, z

