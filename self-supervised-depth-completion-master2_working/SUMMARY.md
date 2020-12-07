
Run main.py --data-folder /home/kamil/Dropbox/Current_research/data/kitti

We use the official error metrics for the KITTI depth completion
benchmark [4], including rmse, mae, irmse, and imae. Specifically, rmse and mae stand
for the root-mean-square error and the mean absolute error, respectively; irmse and imae
stand for the root-mean-square error and the mean absolute error in the inverse depth
representation.
output results:
rmse - 815 (training dziala z 10000 do 1500 w epoce 0)
mae - 230 (traning dziala z 10000 do 900 w epoce 0)

rgb
'/home/kamil/Dropbox/Current_research/data/kitti/data_rgb/train/2011_09_26_drive_0009_sync/image_02/data/0000000286.png',

depth velodyne
'/home/kamil/Dropbox/Current_research/data/kitti/data_depth_velodyne/train/2011_09_26_drive_0001_sync/proj_depth/velodyne_raw/image_03/0000000059.png',

depth gt
'/home/kamil/Dropbox/Current_research/data/kitti/data_depth_annotated/train/2011_09_26_drive_0009_sync/proj_depth/groundtruth/image_03/0000000376.png',

Input

input x is a dict which has three inputs:
d - depth
gt - 
g - 

bathc size is 1

The inputs are
torch.Size([1, 1, 352, 1216])
torch.Size([1, 1, 352, 1216])
torch.Size([1, 1, 352, 1216])


after 1 conv, the inputs are
Out[6]: torch.Size([1, 32, 352, 1216])

and when we return , we get the image back
torch.Size([1, 1, 352, 1216])



conv1.shape
Out[15]: torch.Size([1, 64, 352, 1216])
conv2.shape
Out[16]: torch.Size([1, 64, 352, 1216])
conv3.shape
Out[17]: torch.Size([1, 128, 176, 608])
conv4.shape
Out[18]: torch.Size([1, 256, 88, 304])
conv5.shape
Out[19]: torch.Size([1, 512, 44, 152])
conv6.shape
Out[20]: torch.Size([1, 512, 22, 76])

decoder

torch.Size([1, 768, 44, 152])
torch.Size([1, 384, 88, 304])
torch.Size([1, 192, 176, 608])
torch.Size([1, 128, 352, 1216])
torch.Size([1, 128, 352, 1216])
torch.Size([1, 1, 352, 1216])

depth criteria (in main.py)

  if 'sparse' in args.train_mode:
                depth_loss = depth_criterion(pred, batch_data['d'])
                mask = (batch_data['d'] < 1e-3).float()
            elif 'dense' in args.train_mode:
                depth_loss = depth_criterion(pred, gt)
                mask = (gt < 1e-3).float()

depth_criterion = criteria.MaskedMSELoss() if (
    args.criterion == 'l2') else criteria.MaskedL1Loss()

Choosnf self.input "sparse" or "dense" służy raczej czy spawdzamy output against the sparse input or dense depth gt

*******************************************

COMPARISON BETWEEN run_batch_kitti_pt_line_repro points taken from (n,3) raw lidar point of view points
and png camera points of view from sparse to dense

sparse to dense
print(depth[140][-100:].squeeze())


1)
/home/kamil/Dropbox/Current_research/data/kitti/data_depth_velodyne/train/2011_09_29_drive_0071_sync/proj_depth/velodyne_raw/image_02/0000000223.png

running main_orig.pu 
then iterate()
- for i, batch_data in enumerate(loader):
(and then train_dataset = KittiDepth('train', args))
printed by kitt_loader.py
def depth_read
  
there we get the depth, in this case sparse depth
 sparse = depth_read(self.paths['d'][index]) if (self.paths['d'][index] is not None and self.args.use_d) else None

sparse.shape can be different, e.g. (352, 1216, 1), 
here we transform it to a standard shape (352, 1216, 1)

    if sparse is not None:
        sparse = transform_geometric(sparse)


    img_file = Image.open(filename)
    depth_png = np.array(img_file, dtype=int)
    img_file.close()

    print(depth_png[140][-100:]) #first array
    depth = depth_png.astype(np.float) / 256.
    depth = np.expand_dims(depth, -1)
    print(depth[140][-100:].squeeze()) #second array

    0    0    0    0    0    0    0    0    0    0    0    0 2125    0
    0    0 2126    0    0 2125    0    0    0 2123    0    0    0 2124
    0    0    0 2125 2124    0    0    0 2125    0    0    0 2124    0
    0    0]
[
 0.    0.    8.301 0.    0.    0.    8.305 0.    0.    8.301 0.    0.
 0.    8.293 0.    0.    0.    8.297 0.    0.    0.    8.301 8.297 0.
 0.    0.    8.301 0.    0.    0.    8.297 0.    0.    0.    0.    0.]
 
Note that PIL Image and matplotlib read the images differently

matplotlib gets values

      0.         0.
  0.         0.         0.         0.         0.01264973 0.
  0.         0.         0.         0.         0.00949111 0.
  0.         0.         0.         0.         0.         0.

and Image

0 2125 2124    0    0    0 2125    0    0    0 2124    0
    0    0    0    0    0  

and there is also cv2.imread

 2) running the same image
    kitti_raw = '/home/kamil/Dropbox/Current_research/data/kitti/data_rgb/all/2011_09_29/2011_09_29_drive_0071_sync/velodyne_points/data'

if "2011_09_29_drive_0071_sync" in img_path and "0000000223" in img_path:
     run_batch (degxin)
     inds= np.where((coords_[1] >= 139) & (coords_[1] <= 141) & (coords_[0]>1300) )
     print(coords_[:, inds])
     print(pt_dep[inds])
     
 [[[1360.47583955 1355.4991294  1350.6794501  1345.40876474]]
 [[ 140.00719713  140.12300989  140.38015149  140.94825396]]]
[6.4513655 6.4843674 6.521369  6.623371 ]

---------------------------------------------------------
prinitng lines in

self-supervised-depth-completion-master2/run_batch_kitti_pt_line_repro.py
print_projection_plt
choosing lines there
----------------------------------------------

       device='cuda:0', requires_grad=True)
tensor([ 0, 51, 52, 55, 57, 58, 53, 54, 59, 60, 50, 62, 49, 47, 64, 63, 45, 44,
        56, 43, 48, 46, 61, 40, 42, 38, 39, 33, 31, 36, 35, 27, 23, 26, 25, 41,
        18, 37, 29, 16, 21, 19, 24, 15, 32, 34, 22, 30, 13, 11,  4,  7,  2,  1,
         3,  6, 14, 10,  8,  9, 12, 17, 20,  5, 28], device='cuda:0')


0: 1034
1: 0
2: 0
3: 0
4: 0
5: 0
6: 0
7: 0
8: 0
9: 0
10: 0
11: 0
12: 0
13: 28
14: 0
15: 203
16: 271
17: 0
18: 329
19: 376
20: 0
21: 424
22: 13
23: 452
24: 195
25: 270
26: 468
27: 430
28: 37
29: 452
30: 14
31: 460
32: 103
33: 812
34: 59
35: 426
36: 470
37: 469
38: 465
39: 449
40: 430
41: 178
42: 392
43: 353
44: 413
45: 390
46: 319
47: 466
48: 329
49: 481
50: 409
51: 504
52: 411
53: 445
54: 364
55: 421
56: 286
57: 388
58: 420
59: 406
60: 466
61: 159
62: 431
63: 410150
64: 312