# draws squares over an image
import os

from PIL import Image, ImageDraw
import numpy as np
import time

#batch_data['rgb']

#S_numpy.shape[1]

def draw(type_feature, rgb, depth, switches_2d_argsort, shape_1, run_info, iter=None, result=None):

    print("Drawing")
    hor = switches_2d_argsort % shape_1
    ver = np.floor(switches_2d_argsort // shape_1)
    # print(ver[:10], hor[:10])
    # print("and")
    # print(ver[-10:], hor[-10:])

    # depth back
    depth = depth.squeeze()
    depth_unnor = depth * 256
    #ma2 = Image.fromarray(depth_unnor).convert("RGB")


    ma = rgb.detach().cpu().numpy().squeeze()
    ma = np.transpose(ma, axes=[1, 2, 0])
    # ma = np.uint8(ma)
    ma2 = Image.fromarray(np.uint8(ma)).convert('RGB')
    # create rectangle image
    img1 = ImageDraw.Draw(ma2)

    if type_feature == "sq":
        size = 40

        for ii in range(len(switches_2d_argsort)):
            s_hor = hor[ii]#.detach().cpu().numpy()
            s_ver = ver[ii]#.detach().cpu().numpy()
            # print("Top square switches: ")
            # print(s_ver, s_hor)
            shape = [(s_hor * size, s_ver * size), ((s_hor + 1) * size, (s_ver + 1) * size)]
            # print("shape: ", shape)

            img1.rectangle(shape, outline="red", width=3)

        tim = time.time()
        features_image_path = "/".join(run_info)
        features_image_path = "switches_photos/" + features_image_path
        os.makedirs(features_image_path, exist_ok=True)
        if iter is not None:
            ma2.save(f"{features_image_path}/squares_it_{iter}_{result.rmse:1f}.jpg")
        else:
            ma2.save(f"{features_image_path}/squares_{tim}_{result.rmse:n}.jpg")
        print("saving")
    elif type_feature == "lines":
        print_square_num = 65
        r = 1
        parameter_mask = np.load("kitti_pixels_to_lines_masks.npy", allow_pickle=True)

        # for m in range(10,50):
        #     im = Image.fromarray(parameter_mask[m]*155)
        #     im = im.convert('1')  # convert image to black and white
        #     im.save(f"switches_photos/lala_{m}.jpg")

        for ii in switches_2d_argsort:
            points = parameter_mask[ii]
            y = np.where(points == 1)[0]
            x = np.where(points == 1)[1]
            print(f"Lines {ii} has {len(y)}")

            for p in range(len(x)):
                img1.ellipse((x[p] - r, y[p] - r, x[p] + r, y[p] + r), fill=(255, 0, 0, 0))

        lala = ma2.save(f"switches_photos/lines/lines_{iter}.jpg")
        print("saving")