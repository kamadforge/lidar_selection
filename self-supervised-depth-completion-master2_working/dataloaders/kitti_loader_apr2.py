import os
import os.path
from os.path import dirname
import glob
import fnmatch  # pattern matching
import numpy as np
from numpy import linalg as LA
from random import choice
from PIL import Image
import torch
import torch.utils.data as data
import cv2
import socket
from dataloaders import transforms
from dataloaders.pose_estimator import get_pose_pnp
from scipy.stats import binned_statistic_2d
import matplotlib.pyplot as plt

# binary file read
from run_batch_kitti_pt_line_repro import load_from_bin, velo_points_filter, velo3d_2_camera2d_points, calib_oxts2velo, velo_point_rectify_egomotion, ego_motion_compute_each_lidar_scan

input_options = ['d', 'rgb', 'rgbd', 'g', 'gd']




def load_calib():
    """
    Temporarily hardcoding the calibration matrix using calib file from 2011_09_26
    """
    calib = open("dataloaders/calib_cam_to_cam.txt", "r")
    lines = calib.readlines()
    P_rect_line = lines[25]

    Proj_str = P_rect_line.split(":")[1].split(" ")[1:]
    Proj = np.reshape(np.array([float(p) for p in Proj_str]),
                      (3, 4)).astype(np.float32)
    K = Proj[:3, :3]  # camera matrix

    # note: we will take the center crop of the images during augmentation
    # that changes the optical centers, but not focal lengths
    K[0, 2] = K[
        0,
        2] - 13  # from width = 1242 to 1216, with a 13-pixel cut on both sides
    K[1, 2] = K[
        1,
        2] - 11.5  # from width = 375 to 352, with a 11.5-pixel cut on both sides
    return K


def get_paths_and_transform(split, args):
    assert (args.use_d or args.use_rgb
            or args.use_g), 'no proper input selected'

    if split == "train":
        transform = train_transform
        glob_d = os.path.join(
            args.data_folder,
            'data_depth_velodyne/train/*_sync/proj_depth/velodyne_raw/image_0[2,3]/*.png'
        )
        glob_gt = os.path.join(
            args.data_folder,
            'data_depth_annotated/train/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png'
        )

        def get_rgb_paths(p):
            ps = p.split('/')
            # pnew = '/'.join([args.data_folder] + ['data_rgb'] + ps[-6:-4] +
            #                ps[-2:-1] + ['data'] + ps[-1:])
            pnew = '/'.join([args.data_folder] + ['data_rgb', 'all', ps[-5:-4][0][:10]] + ps[-5:-4] +
                            ps[-2:-1] + ['data'] + ps[-1:])
            return pnew
    elif split == "val":
        if args.val == "full":
            transform = val_transform
            glob_d = os.path.join(
                args.data_folder,
                'data_depth_velodyne/val/*_sync/proj_depth/velodyne_raw/image_0[2,3]/*.png'
            )
            glob_gt = os.path.join(
                args.data_folder,
                'data_depth_annotated/val/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png'
            )
            def get_rgb_paths(p):
                ps = p.split('/')
                pnew = '/'.join(ps[:-7] +  
                    ['data_rgb']+ps[-6:-4]+ps[-2:-1]+['data']+ps[-1:])
                return pnew
        elif args.val == "select":
            transform = no_transform
            glob_d = os.path.join(
                args.data_folder,
                "depth_selection/val_selection_cropped/velodyne_raw/*.png")
            glob_gt = os.path.join(
                args.data_folder,
                "depth_selection/val_selection_cropped/groundtruth_depth/*.png"
            )
            def get_rgb_paths(p):
                return p.replace("groundtruth_depth","image")
    elif split == "test_completion":
        transform = no_transform
        glob_d = os.path.join(
            args.data_folder,
            "depth_selection/test_depth_completion_anonymous/velodyne_raw/*.png"
        )
        glob_gt = None  #"test_depth_completion_anonymous/"
        glob_rgb = os.path.join(
            args.data_folder,
            "depth_selection/test_depth_completion_anonymous/image/*.png")
    elif split == "test_prediction":
        transform = no_transform
        glob_d = None
        glob_gt = None  #"test_depth_completion_anonymous/"
        glob_rgb = os.path.join(
            args.data_folder,
            "depth_selection/test_depth_prediction_anonymous/image/*.png")
    else:
        raise ValueError("Unrecognized split " + str(split))

    if glob_gt is not None:
        # train or val-full or val-select
        paths_d = sorted(glob.glob(glob_d)) 
        paths_gt = sorted(glob.glob(glob_gt)) 
        paths_rgb = [get_rgb_paths(p) for p in paths_gt]
    else:  
        # test only has d or rgb
        paths_rgb = sorted(glob.glob(glob_rgb))
        paths_gt = [None] * len(paths_rgb)
        if split == "test_prediction":
            paths_d = [None] * len(
                paths_rgb)  # test_prediction has no sparse depth
        else:
            paths_d = sorted(glob.glob(glob_d))

    if len(paths_d) == 0 and len(paths_rgb) == 0 and len(paths_gt) == 0:
        raise (RuntimeError("Found 0 images under {}".format(glob_gt)))
    if len(paths_d) == 0 and args.use_d:
        raise (RuntimeError("Requested sparse depth but none was found"))
    if len(paths_rgb) == 0 and args.use_rgb:
        raise (RuntimeError("Requested rgb images but none was found"))
    if len(paths_rgb) == 0 and args.use_g:
        raise (RuntimeError("Requested gray images but no rgb was found"))
    if len(paths_rgb) != len(paths_d) or len(paths_rgb) != len(paths_gt):
        raise (RuntimeError("Produced different sizes for datasets"))

    paths = {"rgb": paths_rgb, "d": paths_d, "gt": paths_gt}
    return paths, transform


def rgb_read(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    # rgb_png = np.array(img_file, dtype=float) / 255.0 # scale pixels to the range [0,1]
    rgb_png = np.array(img_file, dtype='uint8')  # in the range [0,255]
    img_file.close()
    return rgb_png

oheight, owidth = 352, 1216
SIZE=40
bin_ver2 = np.arange(0, oheight, SIZE)
bin_ver2 = np.append(bin_ver2, oheight)
bin_hor2 =np.arange(0, owidth, SIZE)
bin_hor2 = np.append(bin_hor2, owidth)
#global params
params = np.zeros((len(bin_ver2)-1, len(bin_hor2)-1))

values1  = []
np.save("value.npy", values1)

def depth_read(filename, depth_mode, type_feature):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt
    torch.set_printoptions(profile="full")
    np.set_printoptions(precision=3, threshold=100000)
    assert os.path.exists(filename), "file not found: {}".format(filename)
    #print(filename)
    img_file = Image.open(filename)
    depth_png = np.array(img_file, dtype=int)
    img_file.close()
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert np.max(depth_png) > 255, \
        "np.max(depth_png)={}, path={}".format(np.max(depth_png),filename)
    #print(depth_png[140][-100:])
    depth = depth_png.astype(np.float) / 256.
    # depth[depth_png == 0] = -1.
    depth = np.expand_dims(depth, -1)
    #print(depth[140][-100:].squeeze())


    bins = None
    if depth_mode=="sparse":
        #alternative velodyne
        #print(filename)

        #getting path of the bin depth file
        if socket.gethostname()!='kamilblade':
           
            path_main = filename[:31]
            folder1 = filename[58:68]
            folder2 = filename[58:84]
            file = filename[-14:-4]+".bin"
            if "2011" not in folder1: # check workaround for if mode is train or val
                path_main = filename[:31]
                folder1 = filename[83:93]
                folder2 = filename[83:109]
                file = filename[-23:-13] + ".bin"
        else:
        
            path_main = filename[:47]
            folder1 = filename[74:84]
            folder2 = filename[74:100]
            file = filename[-14:-4]+".bin"
            if "2011" not in folder1: # check workaround for if mode is train or val
                path_main = filename[:47]
                folder1 = filename[99:109]
                folder2 = filename[99:125]
                file = filename[-23:-13] + ".bin"

        path_binary = os.path.join(path_main, "data_rgb/all", folder1, folder2, "velodyne_points/data", file)
        #print("bin: ", path_binary)
        path_folder1 = os.path.join(path_main, "data_rgb/all", folder1)
        velo_points = load_from_bin(path_binary)
        v2c_filepath = dirname(dirname(dirname(dirname(path_binary)))) + '/calib_velo_to_cam.txt'
        c2c_filepath = dirname(dirname(dirname(dirname(path_binary)))) + '/calib_cam_to_cam.txt'
        #velo_path = subdir + '/' + file

        # transforming the binary points and getting lines
        mode='02'
        image_rgb = cv2.imread(path_binary)
        xyz_changed = ego_motion_compute_each_lidar_scan(dirname(path_binary), file)
        velo_points_rectified = velo_point_rectify_egomotion(velo_points, xyz_changed)
        # compute depth, line ids and then project them to camera coords from velo coords
        coords_, pt_c, pt_dep, line_id, line_c = velo3d_2_camera2d_points(velo_points_rectified,v_fov=(-24.9, 2.0),h_fov=(-45, 45),vc_path=v2c_filepath,cc_path=c2c_filepath, mode=mode)

        #sieve through the points
        coords_new=[]; line_id_new=[]
        for i in range(len(coords_[0])):
            hor = int(np.floor(coords_[0][i]))
            ver = int(np.floor(coords_[1][i]))
            if hor > 0 and ver > 0 and hor < depth.shape[1] and ver < depth.shape[0]:
                coords_new.append(coords_[:, i]); line_id_new.append(line_id[i])
        coords_new = np.array(coords_new)
        line_id_new = np.array(line_id_new)

        # copying and pruning (last condition)
        #important_lines = [30,  7, 17, 20,  2,  1,  3,  4]
        lines_num = 65
        important_lines = np.arange(lines_num)

        if type_feature=="lines":
            #line adjustment
            # create line_ids vecotor with the same number of points for each line,
            #sample if too many, remove all if too few
            # it has line number if selected and -1 if not
            sel_line_id = np.ones(coords_new.shape[0])*(-1)
            for p in range(lines_num):
                line_id_inds = np.where(line_id_new==p)[0]
                pts_sel = 100
                if len(line_id_inds)>pts_sel:
                    line_id_inds_rand = np.random.choice(line_id_inds, pts_sel, replace=False)
                    sel_line_id[line_id_inds_rand]=p
            print(f"pts depth: {len(np.where(sel_line_id>-1)[0])}")

        #copying points from (num_pts,2) shape to the (y_im, x_im, depth)
        # for both squares and lines
        depth_binary = np.zeros_like(depth)
        for i in range(coords_new.shape[0]):
            # selects from all depth points, so that we have the same num for each line
            if type_feature=="sq" or type_feature=="None" or type_feature=="hlines" or (sel_line_id[i]>-1 and type_feature=="lines"):
                hor = int(np.floor(coords_new[i][0]))
                hor = int(np.floor(coords_new[i][0]))
                ver = int(np.floor(coords_new[i][1]))
                depth_binary[ver, hor] = pt_dep[i]
        depth = depth_binary
        print("sparse depth from binary")
        depth_points = np.where(depth > 0)

        if type_feature=="sq":

            #binning
            size_of_bin = 40
            #print(depth.shape)
            bin_ver=np.arange(0, oheight, size_of_bin)
            bin_ver=np.append(bin_ver, oheight)
            bin_hor=np.arange(0, owidth, size_of_bin)
            bin_hor = np.append(bin_hor, owidth)
            values = depth[np.where(depth > 0)[0], np.where(depth > 0)[1]] # look at pixels with non-0 depth
            #bin function
            bins_2d_depth = binned_statistic_2d(depth_points[0], depth_points[1], values.squeeze(), 'count', bins=[bin_ver, bin_hor], range= [[0, owidth], [0, oheight]])
            #print("bins shape", bins_2d_depth.statistic.shape)

            #saving the bins
            # if os.path.isfile("value.npy"):
            #     values1 = np.load("value.npy", allow_pickle=True)
            # else:
            #     values1  = []
            # if len(values1)==0:
            #     values1=np.expand_dims(bins_2d_depth.statistic, axis=0)
            # else:
            #     values1 = np.concatenate((values1, np.expand_dims(bins_2d_depth.statistic, axis=0)))
            # np.save("value.npy", values1)
            # #print(values1.shape)
            # #with np.printoptions(suppress=True, precision=3):
            # #    print(np.mean(values1, axis=0))
            # del values1

            bins = bins_2d_depth.statistic

            #uncomment!
            #depth = depth_adjustment(depth, depth_points, bins_2d_depth)


        #print(f"Number of depth points: {len(np.where(depth>0)[0])}")

    return depth, bins #375, 1242 #376, 1241


#for all the points in the bin we want to change for a fixed set of points so that each bin has the number equally spaced points
def depth_adjustment(depth, depth_points, bins_2d_depth):
    #depth_points[0] - ver coordinates of posiitve dept points
    # depth_points[1] - hor coordinates of posiitve dept points

    depth_new = np.zeros_like(depth)
    # find the set of points for each bin
    max_bin = 400 #max(bins_2d_depth.binnumber)
    for i in range(max_bin):
        #print("bin", i)
        bin_i_points = np.where(bins_2d_depth[3] == i)[0]
        #print(bin_i_points)

        #if the big bin has at least one points
        if len(bin_i_points)>10:
            #print("points", len(bin_i_points))
            if len(bin_i_points)==1:
                a=5

            # get the corner of the bin square from which the point comes
            corner = [depth_points[0][bin_i_points[0]] - [depth_points[0][bin_i_points[0]] % 40], depth_points[1][bin_i_points[0]] - [depth_points[1][bin_i_points[0]] % 40]]
            bin_sub_ver = np.linspace(corner[0], corner[0]+40, 5 ) #four bins 10 pixels each
            bin_sub_hor = np.linspace(corner[1], corner[1]+40, 5) #four bins 10 pixels each

            # get the coordiantes of the points for each big bin
            depth_sub_points_ver, depth_sub_points_hor, depth_sub =[], [], []
            for i2 in bin_i_points:
                depth_sub_points_ver.append(depth_points[0][i2])
                depth_sub_points_hor.append(depth_points[1][i2])
                #add the depth of the points in that given big bin
                depth_sub.append(depth[depth_points[0][i2], depth_points[1][i2]])

            depth_sub_points_ver = np.array(depth_sub_points_ver).squeeze()
            depth_sub_points_hor = np.array(depth_sub_points_hor).squeeze()
            depth_sub = np.array(depth_sub).squeeze()
            bin_sub_ver = np.array(bin_sub_ver).squeeze()
            bin_sub_hor = np.array(bin_sub_hor).squeeze()



            bins_2d_depth_sub = binned_statistic_2d(depth_sub_points_ver, depth_sub_points_hor, depth_sub, 'mean', bins=[bin_sub_ver, bin_sub_hor])

            depth_sub_new = bins_2d_depth_sub.statistic

            # Find indices that you need to replace
            inds = np.where(np.isnan(depth_sub_new))

            # Place column means in the indices. Align the arrays using take
            depth_sub_new[inds[0], inds[1]]  = np.mean(depth_sub)

            sub_meshgrid = np.meshgrid((bin_sub_ver[:-1] - 5).astype(int), (bin_sub_hor[:-1] - 5).astype(int))

            depth_sub_new = np.expand_dims(depth_sub_new, axis=2)

            # adding the means of the points to the center of the square, doing try because of the remainder part at the end of the image which may not fit
            try:
                #print(sub_meshgrid)
                depth_new[sub_meshgrid] = depth_sub_new
            except:
                print("err")

    #print(len(np.where(depth_new>0)[0]))

    return depth_new





def drop_depth_measurements(depth, prob_keep):
    mask = np.random.binomial(1, prob_keep, depth.shape)
    depth *= mask
    return depth

def train_transform(rgb, sparse, target, rgb_near, args):
    # s = np.random.uniform(1.0, 1.5) # random scaling
    # angle = np.random.uniform(-5.0, 5.0) # random rotation degrees
    do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip

    transform_geometric = transforms.Compose([
        # transforms.Rotate(angle),
        # transforms.Resize(s),
        transforms.BottomCrop((oheight, owidth)),
        transforms.HorizontalFlip(do_flip)
    ])
    if sparse is not None:
        sparse = transform_geometric(sparse)
    target = transform_geometric(target)
    if rgb is not None:
        brightness = np.random.uniform(max(0, 1 - args.jitter),
                                       1 + args.jitter)
        contrast = np.random.uniform(max(0, 1 - args.jitter), 1 + args.jitter)
        saturation = np.random.uniform(max(0, 1 - args.jitter),
                                       1 + args.jitter)
        transform_rgb = transforms.Compose([
            transforms.ColorJitter(brightness, contrast, saturation, 0),
            transform_geometric
        ])
        rgb = transform_rgb(rgb)
        if rgb_near is not None:
            rgb_near = transform_rgb(rgb_near)
    # sparse = drop_depth_measurements(sparse, 0.9)

    return rgb, sparse, target, rgb_near


def val_transform(rgb, sparse, target, rgb_near, args):
    transform = transforms.Compose([
        transforms.BottomCrop((oheight, owidth)),
    ])
    if rgb is not None:
        rgb = transform(rgb)
    if sparse is not None:
        sparse = transform(sparse)
    if target is not None:
        target = transform(target)
    if rgb_near is not None:
        rgb_near = transform(rgb_near)
    return rgb, sparse, target, rgb_near


def no_transform(rgb, sparse, target, rgb_near, args):
    return rgb, sparse, target, rgb_near


to_tensor = transforms.ToTensor()
to_float_tensor = lambda x: to_tensor(x).float()


def handle_gray(rgb, args):
    if rgb is None:
        return None, None
    if not args.use_g:
        return rgb, None
    else:
        img = np.array(Image.fromarray(rgb).convert('L'))
        img = np.expand_dims(img, -1)
        if not args.use_rgb:
            rgb_ret = None
        else:
            rgb_ret = rgb
        return rgb_ret, img


def get_rgb_near(path, args):
    assert path is not None, "path is None"

    def extract_frame_id(filename):
        head, tail = os.path.split(filename)
        number_string = tail[0:tail.find('.')]
        number = int(number_string)
        return head, number

    def get_nearby_filename(filename, new_id):
        head, _ = os.path.split(filename)
        new_filename = os.path.join(head, '%010d.png' % new_id)
        return new_filename

    head, number = extract_frame_id(path)
    count = 0
    max_frame_diff = 3
    candidates = [
        i - max_frame_diff for i in range(max_frame_diff * 2 + 1)
        if i - max_frame_diff != 0
    ]
    while True:
        random_offset = choice(candidates)
        path_near = get_nearby_filename(path, number + random_offset)
        if os.path.exists(path_near):
            break
        assert count < 20, "cannot find a nearby frame in 20 trials for {}".format(
            path_rgb_tgt)

    return rgb_read(path_near)

def draw_features(rgb, bins):
    img_height = 352
    img_width = 1216
    bin_ver = np.arange(0, img_height, 40)
    bin_ver = np.append(bin_ver, img_height)
    bin_hor = np.arange(0, img_width, 40)
    bin_hor = np.append(bin_hor, img_width)




class KittiDepth(data.Dataset):
    """A data loader for the Kitti dataset
    """
    def __init__(self, split, args):
        self.args = args
        self.split = split
        paths, transform = get_paths_and_transform(split, args)
        self.paths = paths
        self.transform = transform
        self.K = load_calib()
        self.threshold_translation = 0.1
        self.type_feature = args.type_feature




    def __getraw__(self, index):
        rgb = rgb_read(self.paths['rgb'][index]) if \
            (self.paths['rgb'][index] is not None and (self.args.use_rgb or self.args.use_g)) else None
        sparse, bins = depth_read(self.paths['d'][index], "sparse", self.type_feature) if \
            (self.paths['d'][index] is not None and self.args.use_d) else None
        target, bins_gt = depth_read(self.paths['gt'][index], "gt", self.type_feature) if \
            self.paths['gt'][index] is not None else None
        rgb_near = get_rgb_near(self.paths['rgb'][index], self.args) if \
            self.split == 'train' and self.args.use_pose else None
        draw_features(rgb, bins)
        return rgb, sparse, target, rgb_near

    def __getitem__(self, index):
        rgb, sparse, target, rgb_near = self.__getraw__(index)
        rgb, sparse, target, rgb_near = self.transform(rgb, sparse, target,
                                                       rgb_near, self.args)
        r_mat, t_vec = None, None
        if self.split == 'train' and self.args.use_pose:
            success, r_vec, t_vec = get_pose_pnp(rgb, rgb_near, sparse, self.K)
            # discard if translation is too small
            success = success and LA.norm(t_vec) > self.threshold_translation
            if success:
                r_mat, _ = cv2.Rodrigues(r_vec)
            else:
                # return the same image and no motion when PnP fails
                rgb_near = rgb
                t_vec = np.zeros((3, 1))
                r_mat = np.eye(3)

        rgb_col = rgb
        rgb, gray = handle_gray(rgb, self.args)
        candidates = {"rgb":rgb, "d":sparse, "gt":target, \
            "g":gray, "r_mat":r_mat, "t_vec":t_vec, "rgb_near":rgb_near}
        items = {
            key: to_float_tensor(val)
            for key, val in candidates.items() if val is not None
        }

        return items

    def __len__(self):
        return len(self.paths['gt'])
