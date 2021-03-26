#!/usr/bin/env python
# coding: utf-8

# # **Project 3D Velodyne points to 2D camera Image**
# ***
#
# ## Dataset
# [KITTI 2011_09_26_drive_0005 dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php?type=city)
# In this code, rectified image is used.
#
# refer to [KITTI Dataset Paper](http://www.cvlibs.net/publications/Geiger2013IJRR.pdf) for the details of data measurement environment
#
# ## Objective
#
# Project Velodyne points(model : HDL-64E) to 2D camera Image.
#

# In[1]


# In[3]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from os.path import dirname, abspath
import argparse
import sys, getopt

def load_from_bin(bin_path):
    obj = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    # ignore reflectivity info
    return obj[:, :3]


# In[5]:


def depth_color(val, min_d=0, max_d=120):
    """
    print Color(HSV's H value) corresponding to distance(m)
    close distance = red , far distance = blue
    """
    np.clip(val, 0, max_d, out=val)  # max distance is 120m but usually not usual
    return (((val - min_d) / (max_d - min_d)) * 120).astype(np.uint8)


def line_color(val, min_d=1, max_d=64):
    """
    print Color(HSV's H value) corresponding to distance(m)
    close distance = red , far distance = blue
    """
    return (((val - min_d) / (max_d - min_d)) * 120).astype(np.uint8)


def in_h_range_points(points, m, n, fov):
    """ extract horizontal in-range points """
    return np.logical_and(np.arctan2(n, m) > (-fov[1] * np.pi / 180), np.arctan2(n, m) < (-fov[0] * np.pi / 180))


def in_v_range_points(points, m, n, fov):
    """ extract vertical in-range points """
    return np.logical_and(np.arctan2(n, m) < (fov[1] * np.pi / 180), np.arctan2(n, m) > (fov[0] * np.pi / 180))


def fov_setting(points, x, y, z, dist, h_fov, v_fov):
    """ filter points based on h,v FOV  """

    if h_fov[1] == 180 and h_fov[0] == -180 and v_fov[1] == 2.0 and v_fov[0] == -24.9:
        return points

    if h_fov[1] == 180 and h_fov[0] == -180:
        return points[in_v_range_points(points, dist, z, v_fov)]
    elif v_fov[1] == 2.0 and v_fov[0] == -24.9:
        return points[in_h_range_points(points, x, y, h_fov)]
    else:
        h_points = in_h_range_points(points, x, y, h_fov)
        v_points = in_v_range_points(points, dist, z, v_fov)
        return points[np.logical_and(h_points, v_points)]


def in_range_points(points, size):
    """ extract in-range points """
    return np.logical_and(points > 0, points < size)


def velo_points_filter(points, angles, v_fov, h_fov):
    """ extract points corresponding to FOV setting """

    # Projecting to 2D
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    depth = np.sqrt(x ** 2)

    if h_fov[0] < -90:
        h_fov = (-90,) + h_fov[1:]
    if h_fov[1] > 90:
        h_fov = h_fov[:1] + (90,)

    x_lim = fov_setting(x, x, y, z, dist, h_fov, v_fov)[:, None]
    y_lim = fov_setting(y, x, y, z, dist, h_fov, v_fov)[:, None]
    z_lim = fov_setting(z, x, y, z, dist, h_fov, v_fov)[:, None]

    # Stack arrays in sequence horizontally
    xyz_ = np.hstack((x_lim, y_lim, z_lim))
    xyz_ = xyz_.T

    # stack (1,n) arrays filled with the number 1
    one_mat = np.full((1, xyz_.shape[1]), 1)
    xyz_ = np.concatenate((xyz_, one_mat), axis=0)

    # need dist info for points color
    dist_lim = fov_setting(dist, x, y, z, dist, h_fov, v_fov)
    depth_lim = fov_setting(depth, x, y, z, dist, h_fov, v_fov)
    angles_lim = fov_setting(angles, x, y, z, dist, h_fov, v_fov)

    color = depth_color(dist_lim, 0, 70)
    angle_c = line_color(angles_lim, 1, 64)

    return xyz_, color, depth_lim, angles_lim, angle_c

# xyz_.shape
# (4, 31861)
# #color.shape
# #(31861,)
# angles_lim.shape
#  (31861, 1)
# depth_lim.shape
# (31861,)
# angle_c.shape
#  (31861, 1)



# In[6]:


def calib_oxts2velo(filepath):
    """
    get Rotation(R : 3x3), Translation(T : 3x1) matrix info
    using R,T matrix, we can convert velodyne coordinates to camera coordinates
    """
    with open(filepath, "r") as f:
        file = f.readlines()

        for line in file:
            (key, val) = line.split(':', 1)
            if key == 'R':
                R = np.fromstring(val, sep=' ')
                R = R.reshape(3, 3)
            if key == 'T':
                T = np.fromstring(val, sep=' ')
                T = T.reshape(3, 1)
    return R, T


def calib_velo2cam(filepath):
    """
    get Rotation(R : 3x3), Translation(T : 3x1) matrix info
    using R,T matrix, we can convert velodyne coordinates to camera coordinates
    """
    with open(filepath, "r") as f:
        file = f.readlines()

        for line in file:
            (key, val) = line.split(':', 1)
            if key == 'R':
                R = np.fromstring(val, sep=' ')
                R = R.reshape(3, 3)
            if key == 'T':
                T = np.fromstring(val, sep=' ')
                T = T.reshape(3, 1)
    return R, T


def calib_cam2cam(filepath, mode):
    """
    If your image is 'rectified image' :
        get only Projection(P : 3x4) matrix is enough
    but if your image is 'distorted image'(not rectified image) :
        you need undistortion step using distortion coefficients(5 : D)

    in this code, I'll get P matrix since I'm using rectified image
    """
    with open(filepath, "r") as f:
        file = f.readlines()

        for line in file:
            (key, val) = line.split(':', 1)
            if key == ('P_rect_' + mode):
                P_ = np.fromstring(val, sep=' ')
                P_ = P_.reshape(3, 4)
                # erase 4th column ([0,0,0])
                P_ = P_[:3, :3]
    return P_


def elevation_angle_compute(points):
    angles = np.zeros((points.shape[0], 1))

    for i in range(points.shape[0]):
        angles[i] = np.arccos(
            (points[i][0] ** 2 + points[i][1] ** 2) / (points[i][0] ** 2 + points[i][1] ** 2 + points[i][2] ** 2))
        if points[i][2] < 0:
            angles[i] = -angles[i]

    return angles / np.pi * 180


def lidar_lineid_compute(angles):
    line_ids = np.zeros((angles.shape[0], 1))
    base_angle = -23.68
    stride = 0.41
    line_num = 64

    for i in range(line_num):
        inds_i = np.where(np.logical_and(angles >= base_angle + i * stride, angles < base_angle + (i + 1) * stride))
        line_ids[inds_i] = i + 1

    return line_ids


def velo3d_2_camera2d_points(points, v_fov, h_fov, vc_path, cc_path, mode='02'):
    """ print velodyne 3D points corresponding to camera 2D image """

    # elevation angle
    angles = elevation_angle_compute(points)

    line_ids = lidar_lineid_compute(angles)

    # R_vc = Rotation matrix ( velodyne -> camera )
    # T_vc = Translation matrix ( velodyne -> camera )
    R_vc, T_vc = calib_velo2cam(vc_path)

    # P_ = Projection matrix ( camera coordinates 3d points -> image plane 2d points )
    P_ = calib_cam2cam(cc_path, mode)

    """
    xyz_v - 3D velodyne points corresponding to h, v FOV in the velodyne coordinates
    c_    - color value(HSV's Hue) corresponding to distance(m)

             [x_1 , x_2 , .. ]
    xyz_v =  [y_1 , y_2 , .. ]   
             [z_1 , z_2 , .. ]
             [ 1  ,  1  , .. ]
    """
    xyz_v, c_, depth, line_ids, line_c = velo_points_filter(points, line_ids, v_fov, h_fov)

    # xyz_.shape
    # (4, 31861)
    # #color.shape /c_
    # #(31861,)
    # angles_lim.shape  /depth
    #  (31861, 1)
    # depth_lim.shape / line_ids
    # (31861,)
    # angle_c.shape / line_c
    #  (31861, 1)

    # len(np.where(line_c == 36)[0])
    # 282
    # len(np.where(line_ids == 20)[0])
    # 282

    """
    RT_ - rotation matrix & translation matrix
        ( velodyne coordinates -> camera coordinates )

            [r_11 , r_12 , r_13 , t_x ]
    RT_  =  [r_21 , r_22 , r_23 , t_y ]   
            [r_31 , r_32 , r_33 , t_z ]
    """
    RT_ = np.concatenate((R_vc, T_vc), axis=1)

    # convert velodyne coordinates(X_v, Y_v, Z_v) to camera coordinates(X_c, Y_c, Z_c)
    for i in range(xyz_v.shape[1]):
        xyz_v[:3, i] = np.matmul(RT_, xyz_v[:, i])

    """
    xyz_c - 3D velodyne points corresponding to h, v FOV in the camera coordinates
             [x_1 , x_2 , .. ]
    xyz_c =  [y_1 , y_2 , .. ]   
             [z_1 , z_2 , .. ]
    """
    xyz_c = np.delete(xyz_v, 3, axis=0)

    # convert camera coordinates(X_c, Y_c, Z_c) image(pixel) coordinates(x,y)
    for i in range(xyz_c.shape[1]):
        xyz_c[:, i] = np.matmul(P_, xyz_c[:, i])

    """
    xy_i - 3D velodyne points corresponding to h, v FOV in the image(pixel) coordinates before scale adjustment
    ans  - 3D velodyne points corresponding to h, v FOV in the image(pixel) coordinates
             [s_1*x_1 , s_2*x_2 , .. ]
    xy_i =   [s_1*y_1 , s_2*y_2 , .. ]        ans =   [x_1 , x_2 , .. ]  
             [  s_1   ,   s_2   , .. ]                [y_1 , y_2 , .. ]
    """
    xy_i = xyz_c[::] / xyz_c[::][2]
    ans = np.delete(xy_i, 2, axis=0)

    """
    width = 1242
    height = 375
    w_range = in_range_points(ans[0], width)
    h_range = in_range_points(ans[1], height)

    ans_x = ans[0][np.logical_and(w_range,h_range)][:,None].T
    ans_y = ans[1][np.logical_and(w_range,h_range)][:,None].T
    c_ = c_[np.logical_and(w_range,h_range)]

    ans = np.vstack((ans_x, ans_y))
    """

    return ans, c_, depth, line_ids, line_c


# In[8]:


def print_projection_cv2(points, color, image):
    """ project converted velodyne points into camera image """

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    for i in range(points.shape[1]):
        cv2.circle(hsv_image, (np.int32(points[0][i]), np.int32(points[1][i])), 2, (int(color[i]), 255, 255), -1)

    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)


def print_projection_plt(points, color, image, line_id):
    """ project converted velodyne points into camera image """

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    for i in range(points.shape[1]):
        if 1: #if line_id[i] in [10, 14, 16, 18, 30, 40, 50, 63]:
            cv2.circle(hsv_image, (np.int32(points[0][i]), np.int32(points[1][i])), 2, (int(color[i]), 255, 255), -1)

    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)


def print_projection_16bit_depth(points, color, h, w):
    """ project converted velodyne points into camera image """

    image = np.zeros((h, w))
    for i in range(points.shape[1]):
        if np.int32(points[0][i]) >= 0 and np.int32(points[0][i]) < w and np.int32(points[1][i]) >= 0 and np.int32(
                points[1][i]) < h:
            image[np.int32(points[1][i]), np.int32(points[0][i])] = color[i] * 256.0
    return np.uint16(image)


def print_projection_8bit_line(points, color, h, w):
    """ project converted velodyne points into camera image """
    image = np.zeros((h, w))
    for i in range(points.shape[1]):
        if np.int32(points[0][i]) >= 0 and np.int32(points[0][i]) < w and np.int32(points[1][i]) >= 0 and np.int32(
                points[1][i]) < h:
            image[np.int32(points[1][i]), np.int32(points[0][i])] = color[i]
    return np.uint8(image)


def compute_timestamps(timestamps_f, file_id):
    # return timestamps in seconds
    with open(timestamps_f) as f:
        timestamps_ = f.readlines()
        # file_id = file[7:10]
        timestamps_ = timestamps_[int(file_id)]
        timestamps_ = timestamps_[11:]
        timestamps_ = np.double(timestamps_[:2]) * 3600 + np.double(timestamps_[3:5]) * 60 + np.double(timestamps_[6:])
    return timestamps_


def load_oxts_location(oxts_f):
    with open(oxts_f) as f:
        data = [list(map(np.double, line.strip().split(' '))) for line in f]
        lat = data[0][0]
        lon = data[0][1]
        alt = data[0][2]
    return lat, lon, alt


def oxts2xyz_convert(oxts_f):
    lat, lon, alt = load_oxts_location(oxts_f)
    er = 6378137
    scale = np.cos(lat * np.pi / 180.0)
    mx = scale * lon * np.pi * er / 180
    my = scale * er * np.log(np.tan((90 + lat) * np.pi / 360))
    mz = alt

    return mx, my, mz


def ego_motion_compute_each_lidar_scan(subdir, file):
    # compute ego-vehicle motion during the lidar scan
    # read timestamps and compute the position change during one complete lidar scane and linearly untwiste all lidar points
    timestamps_start_f = dirname(subdir) + '/timestamps_start.txt'
    timestamps_end_f = dirname(subdir) + '/timestamps_end.txt'

    timestamps_start = compute_timestamps(timestamps_start_f, int(file[:-4]))
    timestamps_end = compute_timestamps(timestamps_end_f, int(file[:-4]))

    # vehicle speed (rotation is not compensentated now, only speed!)
    # read three oxts files, then compute the location of three points, then compute the location of staring position and end position based on linear interpolation
    oxts_f1 = dirname(dirname(subdir)) + '/oxts/data/' + str(int(file[:-4]) - 1).rjust(len(file) - 4, '0') + '.txt'
    oxts_f2 = dirname(dirname(subdir)) + '/oxts/data/' + str(int(file[:-4])).rjust(len(file) - 4, '0') + '.txt'
    oxts_f3 = dirname(dirname(subdir)) + '/oxts/data/' + str(int(file[:-4]) + 1).rjust(len(file) - 4, '0') + '.txt'

    # convert oxts coordinates to x y z coords
    x1, y1, z1 = oxts2xyz_convert(oxts_f1)
    x2, y2, z2 = oxts2xyz_convert(oxts_f2)
    x3, y3, z3 = oxts2xyz_convert(oxts_f3)

    timestamp_oxts_f = dirname(dirname(subdir)) + '/oxts/timestamps.txt'
    timestamp_oxts_1 = compute_timestamps(timestamp_oxts_f, int(file[:-4]) - 1)
    timestamp_oxts_2 = compute_timestamps(timestamp_oxts_f, int(file[:-4]))
    timestamp_oxts_3 = compute_timestamps(timestamp_oxts_f, int(file[:-4]) + 1)

    r_start = (timestamps_start - timestamp_oxts_1) / (timestamp_oxts_2 - timestamp_oxts_1)
    x_s, y_s, z_s = x1 + r_start * (x2 - x1), y1 + r_start * (y2 - y1), z1 + r_start * (z2 - z1)

    r_end = (timestamps_end - timestamp_oxts_2) / (timestamp_oxts_3 - timestamp_oxts_2)
    x_e, y_e, z_e = x2 + r_end * (x3 - x2), y2 + r_end * (y3 - y2), z2 + r_end * (z3 - z2)

    # delta_x, delta_y, delta_z = speed_x * (timestamps_end - timestamps_start) * speed_x, speed_y * (
    #        timestamps_end - timestamps_start), speed_z * (timestamps_end - timestamps_start)

    # convert to velo coordinate
    i2v_filepath = dirname(dirname(dirname(subdir))) + '/calib_imu_to_velo.txt'
    R_iv, T_iv = calib_oxts2velo(i2v_filepath)
    RT_oxts2velo = np.concatenate((R_iv, T_iv), axis=1)

    # convert velodyne coordinates(X_v, Y_v, Z_v) to camera coordinates(X_c, Y_c, Z_c)
    xyz_changed = np.matmul(RT_oxts2velo, np.asarray([x_e, y_e, z_e, 1])) - \
                  np.matmul(RT_oxts2velo, np.asarray([x_s, y_s, z_s, 1]))

    return xyz_changed


def velo_point_rectify_egomotion(velo_points, xyz_changed):
    xyz_delta = xyz_changed / velo_points.shape[0]
    velo_points_rectified = velo_points
    for i in range(velo_points.shape[0]):
        velo_points_rectified[i] = velo_points[i] - i * xyz_delta

    return velo_points_rectified


def main(argv):
    #opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])

    job_id = 0#argv[1]

    totol_workers = 50

    kitti_raw = '/srv/beegfs02/scratch/daid_data/data/kitti/raw/'
    kitti_raw = '/home/kamil/Dropbox/Current_research/data/kitti/data_rgb/all/'
    kitti_raw = '/home/kamil/Dropbox/Current_research/data/kitti/data_rgb/all/2011_09_29/2011_09_29_drive_0071_sync/velodyne_points/data'
    #repro_path = '/srv/beegfs02/scratch/tracezuerich/data/datasets/KITTI/depth_repro_dengxin/'
    mode = '02'
    skipped_fms_start = 5  # the beginning and end 5 frames are skipped, following the official kitti proj
    skipped_fms_end = 5

    for subdir, dirs, files in os.walk(kitti_raw):
        for file in sorted(files):
            if file.endswith('.bin'):

                if int(file[:-4]) >= skipped_fms_end and int(file[:-4]) <= len(files) - 1 - skipped_fms_end:

                    if 1: #if int(file[:-4]) % totol_workers == int(job_id)-1:

                        v2c_filepath = dirname(dirname(dirname(subdir))) + '/calib_velo_to_cam.txt'
                        c2c_filepath = dirname(dirname(dirname(subdir))) + '/calib_cam_to_cam.txt'

                        velo_path = subdir + '/' + file
                        img_path = dirname(dirname(subdir)) + '/' 'image_' + mode + '/' + 'data/' + file.replace('.bin',
                                                                                                                 '.png')
                        print(img_path)
                        if 1:#"2011_09_29_drive_0071_sync" in img_path and "0000000223" in img_path:
                            print("lala")


                            # read lidar pts and image
                            velo_points = load_from_bin(velo_path)
                            np.set_printoptions(threshold=1000000)
                            #print(velo_points)


                            image_rgb = cv2.imread(img_path)

                            xyz_changed = ego_motion_compute_each_lidar_scan(subdir, file)

                            velo_points_rectified = velo_point_rectify_egomotion(velo_points, xyz_changed)

                            # compute depth, line ids and then project them to camera coords from velo coords
                            coords_, pt_c, pt_dep, line_id, line_c = velo3d_2_camera2d_points(velo_points_rectified,
                                                                                              v_fov=(-24.9, 2.0),
                                                                                              h_fov=(-45, 45),
                                                                                              vc_path=v2c_filepath,
                                                                                              cc_path=c2c_filepath, mode=mode)
                            inds= np.where((coords_[1] >= 139) & (coords_[1] <= 141) & (coords_[0]>1300) )
                            print(coords_[:, inds])
                            print(pt_dep[inds])
                            dummy=1

                            ########### png image

                            # path_main = "/home/kamil/Dropbox/Current_research/data/kitti/data_depth_velodyne/train/"
                            # path_file = "2011_09_29_drive_0071_sync"
                            # image = "0000000223.png"
                            #
                            # path=os.path.join(path_main, path_file, "proj_depth/velodyne_raw/image_02", image)
                            # import matplotlib.image as image
                            #
                            # im = image.imread(path)
                            #
                            # from PIL import Image
                            # img_file = Image.open(path)
                            # depth_png = np.array(img_file, dtype=int)
                            # #plt.show(im)
                            #
                            # #print(im)


                        save = True
                        if save:
                            repro_path = "imposed"
                            # # saving pt imposed
                            image = print_projection_plt(points=coords_, color=pt_c, image=image_rgb, line_id =line_id)
                            saving_path = dirname(dirname(subdir)).replace(kitti_raw,
                                                                           repro_path) + '/imposed_depth/velodyne_raw/image_' + mode + '/'
                            if not os.path.exists(saving_path):
                                os.makedirs(saving_path)
                            cv2.imwrite(saving_path + file.replace('.bin', '.png'), image)

                            # saving pt reproject 16 bit
                            image = print_projection_16bit_depth(points=coords_, color=pt_dep, h=image_rgb.shape[0],
                                                                 w=image_rgb.shape[1])
                            saving_path = dirname(dirname(subdir)).replace(kitti_raw,
                                                                           repro_path) + '/proj_depth/velodyne_raw/image_' + mode + '/'
                            if not os.path.exists(saving_path):
                                os.makedirs(saving_path)
                            cv2.imwrite(saving_path + file.replace('.bin', '.png'), image)

                            # saving line imposed 8 bi
                            image = print_projection_plt(points=coords_, color=line_c, image=image_rgb, line_id =line_id)
                            saving_path = dirname(dirname(subdir)).replace(kitti_raw,
                                                                           repro_path) + '/imposed_line/velodyne_raw/image_' + mode + '/'
                            if not os.path.exists(saving_path):
                                os.makedirs(saving_path)
                            cv2.imwrite(saving_path + file.replace('.bin', '.png'), image)

                            # saving line reproject 8 bit
                            image = print_projection_8bit_line(points=coords_, color=line_id, h=image_rgb.shape[0],
                                                               w=image_rgb.shape[1])
                            saving_path = dirname(dirname(subdir)).replace(kitti_raw,
                                                                           repro_path) + '/proj_line/velodyne_raw/image_' + mode + '/'
                            if not os.path.exists(saving_path):
                                os.makedirs(saving_path)
                            cv2.imwrite(saving_path + file.replace('.bin', '.png'), image)
                        # print(file)


# main function

if __name__ == '__main__':
    main(sys.argv[1:])




