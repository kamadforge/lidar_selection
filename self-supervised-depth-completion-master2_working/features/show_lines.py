import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision.models import resnet
import matplotlib.pyplot as plt
from torch.nn.parameter import Parameter

torch.set_printoptions(profile="full")

def save_pic(pic, i=None):
    fig = plt.figure(frameon=False)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)


    ax.imshow(pic, cmap='Greys', aspect='auto')
    #ax.imshow(mins[i], cmap='Greys', aspect='auto')
    fig.savefig(f'figures/line_{i}.png')


def process_lines():

    lines=np.load("kitti_pixels_to_lines_proc.npy")


    # # check for duplicate points:
    # for i1 in range(lines.shape[1]):
    #     for i2 in range(lines.shape[2]):
    #         aa = np.where(lines[:, i1, i2]==1)[0]
    #         print(aa)
    #         if len(aa)>1:
    #             elem = np.random.choice(aa, len(aa)-1)
    #             lines[aa, i1, i2]=0

    # find upper and lower boundaries
    mins = np.zeros((lines.shape[0], lines.shape[2]))
    maxes = np.zeros((lines.shape[0], lines.shape[2]))
    minss= {}; maxess = {}
    for i in range(lines.shape[0]):
        minss[i]=[]
        maxess[i]=[]

    for l in range(lines.shape[0]):
        for i2 in range(lines.shape[2]):
            if np.sum(lines[l, :, i2])>0:
                #print(lines[l, :, i2])
                nonzero= np.where(lines[l,:,i2]>0)[0]
                mins[l, i2] = min(nonzero)
                maxes[l, i2] = max(nonzero)

                minss[l].append([i2, min(nonzero)])
                maxess[l].append([i2, min(nonzero)])



    #np.save("kitti_pixels_to_lines_proc.npy", lines)
    #pdrawing new lines
    lines_boundaries=np.zeros_like(lines)
    for l in range(lines.shape[0]):
        # points that are already there
        # for i2 in range(lines.shape[2]):
        #     lines_new[l, int(maxes[l,i2]), i2]=1
        # line of best fit
        if len(minss[l])>5:
            coeffs_min = np.polyfit(np.array(minss[l])[:, 0], np.array(minss[l])[:, 1], 2)
            coeffs_max = np.polyfit(np.array(maxess[l])[:, 0], np.array(maxess[l])[:, 1], 2)
            x = np.arange(lines.shape[2])
            y = coeffs_min[0] * (x ** 2) + coeffs_min[1] * x + coeffs_min[2]
            y_max = coeffs_max[0] * (x ** 2) + coeffs_max[1] * x + coeffs_max[2]
            for i2 in range(lines.shape[2]): #i2 and x same
                print(l, int(y[i2]), i2)
                try:
                    lines_boundaries[l, int(y[i2]), i2]=1
                except IndexError:
                    print("no index")

    lines_new = np.zeros_like(lines)

    for l in range(1, lines.shape[0]):
        print(l)
        if l==64:
            dd=3
            print("dd")
        for i2 in range(lines.shape[2]):
            #print("i2: ", i2)
            # if we have upper bound of a line region given by the next line
            if l+1< lines.shape[0]:
                v_up = np.where(lines_boundaries[l+1, :, i2]==1)[0]
            v_down = np.where(lines_boundaries[l, :, i2]==1)[0]
            if len(v_down)==0:
                v_d=lines.shape[1]-1 #down boundary
            else:
                v_d = v_down[0]
            if len(v_up)>0:
                v_u=v_up[0];
                while v_u < v_d:
                    lines_new[l, v_u, i2]=1
                    v_u = v_u + 1

    np.save("kitti_pixels_to_lines_masks.npy", lines_new)


    for i in range(65):
        save_pic(lines_new[i])

    # plt.imshow(lines[35])
    # plt.savefig('figures/line_35.png')
    # plt.show()


    parameter_mask = torch.tensor(lines)


    parameter = Parameter(-1e-10 * torch.ones(65), requires_grad=True)
    phi = F.softplus(parameter)


    if any(torch.isnan(phi)):
        print("some Phis are NaN")
    # it looks like too large values are making softplus-transformed values very large and returns NaN.
    # this occurs when optimizing with a large step size (or/and with a high momentum value)


    S = phi / torch.sum(phi)


    S_mask_ext = torch.einsum("i, ijk->ijk", [S, parameter_mask])
    print(S_mask_ext[24][308][733])
    S_mask=torch.max(S_mask_ext, 0)[0]
    S_mask_ind=torch.max(S_mask_ext, 0)[1]
    #print(S_mask_ind[300])

    np.set_printoptions(suppress=True)
    point_nums=np.zeros(65,)
    for i in range(len(parameter)):
        print(f"{i}: {len(np.where(S_mask_ind==i)[0])}")
        point_nums[i] = len(np.where(S_mask_ind==i)[0])

    print(np.argsort(point_nums))
    print(np.sort(point_nums))

    a=0