# This function is used to create a mask which can be used for evaluating the model on the subset of features


import numpy as np

from scipy.stats import binned_statistic_2d

from depth_draw import draw

# print("bins shape", bins_2d_depth.statistic.shape)


#for all the points in the bin we want to change for a fixed set of points so that each bin has the number equally spaced points
def depth_adjustment(depth, adjust, rgb=None):
    #depth_points[0] - ver coordinates of posiitve dept points
    # depth_points[1] - hor coordinates of posiitve dept points

    depth = depth.detach().cpu().numpy().squeeze()
    depth_points = np.where(depth > 0)

    # binning
    size_of_bin = 40
    oheight = depth.shape[0]
    owidth = depth.shape[1]
    bin_ver = np.arange(0, oheight, size_of_bin)
    bin_ver = np.append(bin_ver, oheight)
    bin_hor = np.arange(0, owidth, size_of_bin)
    bin_hor = np.append(bin_hor, owidth)
    values = depth[np.where(depth > 0)[0], np.where(depth > 0)[1]]  # look at depth values for pixels with non-0 depth
    square_num = (len(bin_ver)-1)*(len(bin_hor)-1)

    # bin function
    bins_2d_depth = binned_statistic_2d(depth_points[0], depth_points[1], values.squeeze(), 'count',
                                        bins=[bin_ver, bin_hor], range=[[0, owidth], [0, oheight]])
    # hash bin function each bin number has a list of points that belong to that bin
    bin_hash = {}
    for b in range(square_num):
        bin_hash[b] = np.where(bins_2d_depth.binnumber == b)



    # counting points
    A_this = bins_2d_depth.statistic
    if 'A' not in globals():
        global A;
        A=A_this
    else:
        A=A_this+A

    print("A and points")
    print(A[7,3])
    print(A_this[7,3])

    A_2d_sort = np.sort(A, None)
    A_2d_argsort = np.argsort(A, None)
    ver = np.floor(A_2d_argsort // A.shape[1])
    hor = A_2d_argsort % A.shape[1]

    coord_sorted = np.stack([ver, hor]).transpose()
    print("Squares by the most points")

    print(coord_sorted[-10:])
    print(A_2d_argsort[-10:])
    print("num of points: ", A_2d_sort[-10:])

    # choose ranks for the squares
    select_mask=True # to create a mask with 1s for selected squates and 0 otherwise
    squares = np.arange(square_num)
    sq_mode = "switch"
    if sq_mode == "random":
        np.random.seed(15)
        squares = np.random.choice(square_num, 10)
    elif sq_mode == "most":
        squares = np.array([int(a) for a in A_2d_argsort[-10:]])
    elif sq_mode == "switch":
        squares = np.load(f"ranks/switches_argsort_2D_iter_507.npy")[-10:]
    print(f"Squares used {sq_mode}: ", squares)

    # draw the selected squares
    if rgb != None and 0:
        draw("sq", rgb, squares, A.shape[1])

    ver = np.floor(squares // A.shape[1])
    hor = squares % A.shape[1]
    if select_mask:
        mask_new = np.zeros_like(depth)
        for p, item in enumerate(squares):
            verp_scaled= int(ver[p])*size_of_bin
            horp_scaled=hor[p]*size_of_bin
            mask_new[verp_scaled:(verp_scaled+size_of_bin), horp_scaled:(horp_scaled+size_of_bin)]=1

        depth=mask_new*depth


    remove_depth=0
    if remove_depth:
        #depth = np.random.normal(10, 5, (depth.shape[0], depth.shape[1])) #RMSE=21186.322, MAE=19962.695
        depth =np.zeros_like(depth)

    points_wanted_num = 50
    points_choice = "random"
    print(f"adjust: {adjust}, points_wanted_num: {points_wanted_num}, points_choice: {points_choice}")

    if adjust:
        depth_new = np.zeros_like(depth)
        # find the set of points for each bin
        max_bin = 400 #max(bins_2d_depth.binnumber)
        table_i=np.zeros(max_bin)
        for i in range(max_bin):
            #print("bin", i)
            bin_i_points = np.where(bins_2d_depth[3] == i)[0]
            #print(bin_i_points)
            #if the big bin has at least one points
            if len(bin_i_points)>points_wanted_num:
                table_i[i]+=1
                if points_choice == "random":
                    points_chosen = np.random.choice(bin_i_points, points_wanted_num, replace=False)
                    for j in points_chosen:
                        depth_new[depth_points[0][j], depth_points[1][j]] = depth[depth_points[0][j], depth_points[1][j]]

                else: #grid of points
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

        #print(table_i)

        depth = depth_new

    return depth


def depth_adjustment_lines(depth):
    # line adjustment
    # create line_ids vecotor with the same number of points for each line,
    # sample if too many, remove all if too few
    # it has line number if selected and -1 if not
    sel_line_id = np.ones(coords_new.shape[0]) * (-1)
    for p in range(lines_num):
        line_id_inds = np.where(line_id_new == p)[0]
        pts_sel = 100
        if len(line_id_inds) > pts_sel:
            line_id_inds_rand = np.random.choice(line_id_inds, pts_sel, replace=False)
            sel_line_id[line_id_inds_rand] = p
    print(f"pts depth: {len(np.where(sel_line_id > -1)[0])}")

    # copying points from (num_pts,2) shape to the (y_im, x_im, depth)
    # for both squares and lines
    depth_binary = np.zeros_like(depth)
    for i in range(coords_new.shape[0]):
        # selects from all depth points, so that we have the same num for each line
        if type_feature == "sq" or type_feature == "None" or (sel_line_id[i] > -1 and type_feature == "lines"):
            hor = int(np.floor(coords_new[i][0]))
            hor = int(np.floor(coords_new[i][0]))
            ver = int(np.floor(coords_new[i][1]))
            depth_binary[ver, hor] = pt_dep[i]
    depth = depth_binary
    depth_points = np.where(depth > 0)

