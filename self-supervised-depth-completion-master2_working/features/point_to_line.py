import os

from PIL import  Image
import numpy as np
import scipy.spatial
#from show_lines import save_pic
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob

import sys
sys.path.insert(0,'..')
from dataloaders import transforms


# save a 0-1 file as image (to show line)
def save_pic(pic, i=None):
    fig = plt.figure(frameon=False)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(pic, cmap='Greys', aspect='auto')
    #ax.imshow(mins[i], cmap='Greys', aspect='auto')
    fig.savefig(f'../figures/line_{i}.png')

# creates a dic of points (hor, ver coordinates) which belong to each line (not all points in the rgb image are part of it, only depth points that are in the depth image)
def separate_dic_from_gtpoints(segmented_line_file=None):
    #input: vaishakh's line numbers (some curve which is deemed to be a line)

    if segmented_line_file is None:
        im_lines = Image.open("/home/kamil/Dropbox/Current_research/data/kitti/line_segmentation/train/segmented_lines/2011_09_26/2011_09_26_drive_0001_sync/lines_segmented_02/data/0000000006.png")
    else:
        im_lines = Image.open(segmented_line_file)
    image_lines = np.array(im_lines)


    oheight = 352
    owidth = 1216

    transform_geometric = transforms.Compose([
        # transforms.Rotate(angle),
        # transforms.Resize(s),
        transforms.BottomCrop((oheight, owidth)),
        #transforms.HorizontalFlip(do_flip)
    ])

    image_lines = transform_geometric(image_lines)

    # save cropped debug (the image we just read, slightly cropped and clearer)
    im2 = Image.fromarray(np.uint8(image_lines * 255))
    im2.save("segmented_lines.png")


    # we read the depth line, where each pixel value is the number to which line the point belongs to, 1-64
    # we put all the points for line i into one dictionary entry
    # we get a dic of length 64, which contains hor and ver values for each pixel in a given line
    # this takes less space than storing the whole 65,352,1216 binary matrix
    dic_file = {}
    for i in range(1, 65):
        #print(i)
        ver = np.where(image_lines == i)[0]
        #print(len(ver))
        # if len(ver > 0):
        #     lsas = 9
        hor = np.where(image_lines == i)[1]
        #lines_mat[i, ver, hor] = 1
        dic_file[i] = [ver, hor]

    highest_depth_point = np.argmax(image_lines, axis=0)
    for i in range(image_lines.shape[1]):

        h = [i]*highest_depth_point[i]
        v = np.arange(highest_depth_point[i])
        dic_file[64][0] = np.hstack((dic_file[64][0],v))
        dic_file[64][1] = np.hstack((dic_file[64][1],h))

    #np.save("test_im.npy", dic_file)
    return dic_file

####

global_compute=1
global map
global_map = np.zeros((65,352,1216))


def assign_unassigned_pts(global_map):

    global_map_new = np.zeros_like(global_map)

    points_flattened = np.sum(global_map, axis=0)
    unassigned_points = np.where(points_flattened == 0)
    dic_file = {}
    for i in range(1,65):
        dic_file[i]=np.where(global_map[i]>0)


    #we go over the depth points and assign the closest line to the point (e.g. 18000 points)
    for i1 in range(352):
        for i2 in range(1216):
            if points_flattened[i1][i2]==0:
                if i % 1000 == 0:
                    print(i)
                best = [np.inf, np.inf, np.inf]  # the closent line, distance, pixel where it's closest distance]
                for line_num in range(1, 65):
                    line, points = dic_file[line_num]
                    point = [i1, i2]

                    if len(points)>0:
                        # for each depth point compute the distance to the given line points
                        la = scipy.spatial.distance.cdist(np.array(point).reshape((1,2)), np.swapaxes(np.array(dic_file[line_num]), 0, 1))
                        #among all the hitherto lines find the one whose minimum distance to the given point is the smallest
                        if la[0][np.argmin(la)] < best[1]:
                            best = [line_num, la[0][np.argmin(la)], np.argmin(la)]
                global_map_new[best[0], i1, i2] = 1
                print(f"({i1},{i2}) assigned to {best[0]}")
            else:
                line_max = np.argmax(global_map[:, i1, i2])
                global_map_new[line_max, i1, i2]=1
                print(f"({i1},{i2}) already at {line_max}")




    #depth_points_line[i]=int(best[0]) # assigning the line_num to the point
    #print("Line assigned: ", depth_points_line[i])

    unassigned_points_new = np.where(points_flattened == 0)
    print("No points assigned: ", len(unassigned_points_new[0]))

    return global_map_new

def global_mask():
    global_map = np.zeros((65, 352, 1216))
    points_used=352*1216

    path="/home/kamil/Dropbox/Current_research/data/kitti/line_segmentation/train/segmented_lines_dic"

    result = [y for x in os.walk(path) for y in glob(os.path.join(x[0], '*.npy'))]
    print("Number of npy individual file masks: ", len(result))

    for file in result:

        masks_dic = np.load(file, allow_pickle=True)
        # just vashakh's points
        # masks_dic = dic_file CH

        #print("Masks length: ", len(masks_dic))


        # m = Image.open(path_lines)
        # m_all_arr = np.array(m)
        # m_arr=m_all_arr[-352:, -1216:]
        # print(m_arr.shape)
        dic_file = {}
        for i in range(1, 65):
            inds = np.where(masks_dic[:, 2] == i)[0]

            # ver = masks_dic[i][0]
            # hor = masks_dic[i][1]

            ver = masks_dic[inds][:, 0]
            hor = masks_dic[inds][:, 1]

            if len(ver) != 0:
                #masks[i, ver, hor] = 1
                global_map[i, ver, hor] += 1
            # dic_file[i]=[ver, hor]


        print("No points assigned: ", len(np.where(np.sum(global_map, axis=0)==0)[0]))
        #print(points_used - len(np.where(np.sum(global_map, axis=0)==0)[0]))
        points_used = len(np.where(np.sum(global_map, axis=0)==0)[0])

    #visualize(global_map=global_map, global_compute=1)
    global_map = assign_unassigned_pts(global_map)
    np.save("kitti_globalmap_jun22.npy", global_map)



#######
# masks_from_poits for single, global_map for all/global
def visualize(masks_from_points=None, global_map=None, global_compute=0):

    for i in range(65):
        print(i)
        lines_viz = np.zeros((352, 1216))  # empty image

        if global_compute:
            line_max_points = np.where(np.argmax(global_map, axis=0)==i)
            #points = np.where(masks_from_points[:, 2] == i)  # find points belonigng to line i
            for pt in range(len(line_max_points[0])):
                lines_viz[int(line_max_points[0][pt]), int(line_max_points[1][pt])]=1
        else:

            points = np.where(masks_from_points[:, 2] == i)[0] # find points belonigng to line i
            points_on_the_line = masks_from_points[points, :]
            points_on_the_line_y_x = points_on_the_line[:, :2]
            for pt in points_on_the_line_y_x:
                lines_viz[int(pt[0]), int(pt[1])]=1

        plt.imshow(lines_viz)
        plt.savefig(f"lines/line_{i}.png")

        # im = Image.fromarray(lines_viz)
        # im.save(f"../figures/lines_local/line_{i}.png")
        #

    print("fin")

########

def create_mask_from_points(dic_file, depth_file=None, local_mask_path=None):

    if depth_file is None:
        depth_file = "/home/kamil/Dropbox/Current_research/data/kitti/data_depth_velodyne/train/2011_09_26_drive_0001_sync/proj_depth/velodyne_raw/image_02/0000000006.png"
        im_depth = Image.open(depth_file)
    else:
        im_depth = Image.open(depth_file)

    image_depth = np.array(im_depth)
    image_depth_line_assign = np.zeros_like(image_depth)

    print(f"sparse read: {len(np.where(image_depth)[0] > 0)}")

    #image_depth = image_depth[-352:, -1216:]

    oheight = 352
    owidth = 1216

    transform_geometric = transforms.Compose([
        # transforms.Rotate(angle),
        # transforms.Resize(s),
        transforms.BottomCrop((oheight, owidth)),
        #transforms.HorizontalFlip(do_flip)
    ])

    image_depth = transform_geometric(image_depth)

    print(f"sparse read after crop: {len(np.where(image_depth)[0] > 0)}")

    depth_points = np.where(image_depth>0)

    depth_points_line = np.zeros(len(depth_points[0])).astype(int)

    #we go over the depth points and assign the closest line to the point (e.g. 18000 points)
    for i in range(len(depth_points[0])):
        if i % 1000 == 0:
            print(i)
        best = [np.inf, np.inf, np.inf]  # the closent line, distance, pixel where it's closest distance]
        for line_num in range(1, 65):
            line, points = dic_file[line_num]
            point = [depth_points[0][i], depth_points[1][i]]

            if len(points)>0:
                # for each depth point compute the distance to the given line points
                la = scipy.spatial.distance.cdist(np.array(point).reshape((1,2)), np.swapaxes(np.array(dic_file[line_num]), 0, 1))
                #among all the hitherto lines find the one whose minimum distance to the given point is the smallest
                if la[0][np.argmin(la)] < best[1]:
                    best = [line_num, la[0][np.argmin(la)], np.argmin(la)]

                #print(best)

        depth_points_line[i]=int(best[0]) # assigning the line_num to the point
        #print("Line assigned: ", depth_points_line[i])

        # global
        if global_compute:
            global_map[int(best[0]), depth_points[0][i], depth_points[1][i]] += 1  # assigning the line_num to the point

    # 3-dim array with x and y coordinates of a depth point and the line it belongs to (e.g. 18000, 18000, 18000)
    masks_from_points = np.array([depth_points[0], depth_points[1], depth_points_line]).swapaxes(0,1)
    #np.save("test.npy", masks_from_points)

    #local_mask_path = depth_file[:-4]+"_locmask"+".npy"
    np.save(local_mask_path, masks_from_points)
    print(f"saved to {local_mask_path}")

    # visualize a given line
    visialize_depth_line_assignment=0
    if visialize_depth_line_assignment:
        visualize(masks_from_points)

    return local_mask_path

#print(image_depth[200:210])

if __name__=="__main__":
    #dic_file = separate_dic_from_gtpoints()
    #create_mask_from_points(dic_file)
    global_mask()