import argparse
import os
import time
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from dataloaders.kitti_loader_apr2 import load_calib, oheight, owidth, input_options, KittiDepth
from model import DepthCompletionNet
from metrics import AverageMeter, Result
import criteria
import helper
from inverse_warp import Intrinsics, homography_from
from scipy.stats import binned_statistic_2d
import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser(description='Sparse-to-Dense')
parser.add_argument('-w',
                    '--workers',
                    default=0,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs',
                    default=11,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run (default: 11)')
parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-c',
                    '--criterion',
                    metavar='LOSS',
                    default='l2',
                    choices=criteria.loss_names,
                    help='loss function: | '.join(criteria.loss_names) +
                    ' (default: l2)')
parser.add_argument('-b',
                    '--batch-size',
                    default=1,
                    type=int,
                    help='mini-batch size (default: 1)')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=1e-5,
                    type=float,
                    metavar='LR',
                    help='initial learning rate (default 1e-5)')
parser.add_argument('--weight-decay',
                    '--wd',
                    default=0,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 0)')
parser.add_argument('--print-freq',
                    '-p',
                    default=10,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--resume',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--data-folder',
                    default='../data',
                    type=str,
                    metavar='PATH',
                    help='data folder (default: none)')
parser.add_argument('-i',
                    '--input',
                    type=str,
                    default='rgbd',
                    choices=input_options,
                    help='input: | '.join(input_options))
parser.add_argument('-l',
                    '--layers',
                    type=int,
                    default=34,
                    help='use 16 for sparse_conv; use 18 or 34 for resnet')
parser.add_argument('--pretrained',
                    action="store_true",
                    help='use ImageNet pre-trained weights')
parser.add_argument('--val',
                    type=str,
                    default="select",
                    choices=["select", "full"],
                    help='full or select validation set')
parser.add_argument('--jitter',
                    type=float,
                    default=0.1,
                    help='color jitter for images')
parser.add_argument(
    '--rank-metric',
    type=str,
    default='rmse',
    choices=[m for m in dir(Result()) if not m.startswith('_')],
    help='metrics for which best result is sbatch_datacted')
parser.add_argument(
    '-m',
    '--train-mode',
    type=str,
    default="dense",
    choices=["dense", "sparse", "photo", "sparse+photo", "dense+photo"],
    help='dense | sparse | photo | sparse+photo | dense+photo')
parser.add_argument('-e', '--evaluate', default='', type=str, metavar='PATH')
parser.add_argument('--cpu', action="store_true", help='run on cpu')
parser.add_argument('--type_feature', default="None", choices=["sq", "lines", "None"])
parser.add_argument('--training_sparse_opt', default="latin")
parser.add_argument('--feat_num', default=10, type=int)


args = parser.parse_args()
args.use_pose = ("photo" in args.train_mode)
# args.pretrained = not args.no_pretrained
args.result = os.path.join('..', 'results')
args.use_rgb = ('rgb' in args.input) or args.use_pose
args.use_d = 'd' in args.input
args.use_g = 'g' in args.input

#os.environ["CUDA_VISIBLE_DEVICES"]=""
#torch.cuda.is_available = lambda : False

if args.use_pose:
    args.w1, args.w2 = 0.1, 0.1
else:
    args.w1, args.w2 = 0, 0
print(args)

cuda = torch.cuda.is_available() and not args.cpu
print(torch.cuda.get_device_name(torch.cuda.current_device()))
if cuda:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("=> using '{}' for computation.".format(device))



# define loss functions
depth_criterion = criteria.MaskedMSELoss() if (
    args.criterion == 'l2') else criteria.MaskedL1Loss()
photometric_criterion = criteria.PhotometricLoss()
smoothness_criterion = criteria.SmoothnessLoss()

if args.use_pose:
    # hard-coded KITTI camera intrinsics
    K = load_calib()
    fu, fv = float(K[0, 0]), float(K[1, 1])
    cu, cv = float(K[0, 2]), float(K[1, 2])
    kitti_intrinsics = Intrinsics(owidth, oheight, fu, fv, cu, cv)
    if cuda:
        kitti_intrinsics = kitti_intrinsics.cuda()


def iterate(mode, args, loader, model, optimizer, logger, epoch):
    block_average_meter = AverageMeter()
    average_meter = AverageMeter()
    meters = [block_average_meter, average_meter]

    # switch to appropriate mode
    assert mode in ["train", "val", "eval", "test_prediction", "test_completion"], \
        "unsupported mode: {}".format(mode)
    if mode == 'train':
        model.train()
        lr = helper.adjust_learning_rate(args.lr, optimizer, epoch)
    else:
        model.eval()
        lr = 0

    print("\nTraining")
    prune_type = "sq"  # sq, vlines, nothing
    square_choice = args.training_sparse_opt
    if prune_type=="sq":
        print(f"Features: squares\n Square choice: {square_choice}")

    for i, batch_data in enumerate(loader):
        start = time.time()
        batch_data = {
            key: val.to(device)
            for key, val in batch_data.items() if val is not None
        }
        gt = batch_data[
            'gt'] if mode != 'test_prediction' and mode != 'test_completion' else None
        data_time = time.time() - start
        start = time.time()
        

        if prune_type == "vlines":
            np.random.seed(10)
            lines_unmasked=np.random.choice(352,20, replace=False)
            lines_unmasked = np.arange(352)
            lines_all=np.arange(352)
            lines_masked = [x for x in lines_all if x not in lines_unmasked]
            batch_data['d'][:, :, lines_masked]=0
            print(batch_data['d'].shape)
            print("lines unmasked", lines_unmasked)


        elif prune_type == "sq":
            A = np.load("ranks/switches_2D_equal_iter_390.npy", allow_pickle=True)
            # with np.printoptions(precision=5):
            #     print("switches", A)
            #get the ver and hor coordinates of the most important squares
            A_2d_argsort = np.argsort(A, None)[::-1]
            if square_choice=="most":
                squares_top_file = "ranks/sq/global/squares_most.npy"
                A_2d_argsort = np.load(squares_top_file)[::-1]
            ver = np.floor(A_2d_argsort // A.shape[1])
            hor = A_2d_argsort % A.shape[1]
            A_list = np.stack([ver, hor]).transpose()
            square_size = 40
            squares_top_num = args.feat_num

            if square_choice=="full":
                squares_top = A_list

            if square_choice=="most":
                squares_top = A_list[:squares_top_num]

            if square_choice=="best_sw":
                squares_top = A_list[:squares_top_num]

            if square_choice=="latin_sw":
                #creating latin grid (with big squares/blocks)
                hor_large = np.linspace(0,30,7)
                ver_larger = np.arange(10)
                all_squares = np.arange(len(A_list))
                bins_2d_latin= binned_statistic_2d(ver, hor, all_squares, 'min', bins=[ver_larger, hor_large])
                bins_2d_latin.statistic
                best_latin  = bins_2d_latin.statistic[-3:].flatten().astype(int)
                best_latin_coors = list(A_list[best_latin])
                for i1 in A_list:
                    elem_in = False #check if the block already contains a small square
                    for i2 in best_latin_coors:
                        if i1[0]==i2[0] and i1[1]==i2[1] :
                            elem_in = True
                    if not elem_in:
                        best_latin_coors.append(i1)
                    if len(best_latin_coors)==squares_top_num:
                        break;
                squares_top = np.array(best_latin_coors)

            elif square_choice=="latin":
                np.random.seed(12)
                squares_latin_evenlyspaced = []
                # create blocks, choose k random blocks and have fixed first block
                hor_large = np.linspace(0, 30, 7)
                ver_large = np.arange(10)
                # random sample from positive blocks
                hor_large_rand = np.random.choice(len(hor_large), squares_top_num)
                ver_large_rand  = np.random.choice([6,7,8], squares_top_num)
                # selecting a small square from A_list with given corrdinates within a block
                for j in range(len(hor_large_rand)):
                    elem = np.where((A_list[:, 0]== ver_large_rand[j]) & (A_list[:, 1] == hor_large[hor_large_rand[j]]))[0][0]
                    squares_latin_evenlyspaced.append(elem)
                squares_top = A_list[squares_latin_evenlyspaced]


            elif square_choice=="random_all":
                np.random.seed(12)
                rand_idx = np.random.choice(len(A_list), squares_top_num)
                print(rand_idx)
                squares_top = A_list[rand_idx]

            elif square_choice=="random_pos": # from squares which include depth points
                np.random.seed(12)
                #choose from the squares which have roughly positive number of depth points
                rand_idx = np.random.choice(len(A_list[:93]), squares_top_num)
                print(rand_idx)
                squares_top = A_list[rand_idx]

            # after selecting indices of the squares save in squares_top
            squares_top_scaled = np.array(squares_top)* square_size
            mask = np.zeros((352, 1216))
            bin_ver = np.arange(0, 352, square_size)
            bin_ver = np.append(bin_ver, oheight)
            bin_hor = np.arange(0, 1216, square_size)
            bin_hor = np.append(bin_hor, owidth)
            # filling in the mask with selected squares up to squares_top_num (e.g. 20)
            #print("Number of squares selected: ", len(squares_top))
            #print(squares_top)
            for it in range(len(squares_top)): #in all but full should be equal to squares_top_num
                ver = int(squares_top[it][0])
                hor = int(squares_top[it][1])
                #print("ver", bin_ver[ver], bin_ver[ver+1], "hor", bin_hor[hor], bin_hor[hor+1] )
                mask[bin_ver[ver]:bin_ver[ver+1], bin_hor[hor]:bin_hor[hor+1]]=1

            aaa1 = batch_data['d'].detach().cpu().numpy()
            batch_data['d']=torch.einsum("abcd, cd->abcd", [batch_data['d'], torch.Tensor(mask).to(device)])
            aaa2 = batch_data['d'].detach().cpu().numpy()
            #
            # # from PIL import Image
            # # img = Image.fromarray(aaa[0, :, :, :], 'RGB')
            # # #img.save('my.png')
            # # img.show()


        pred = model(batch_data)
        #im = batch_data['d'].detach().cpu().numpy()
        #im_sq = im.squeeze()
        #plt.figure()
        #plt.imshow(im_sq)
        #plt.show()
        #for i in range(im_sq.shape[0]):
        #    print(f"{i} - {np.sum(im_sq[i])}")

        depth_loss, photometric_loss, smooth_loss, mask = 0, 0, 0, None
        if mode == 'train':
            # Loss 1: the direct depth supervision from ground truth label
            # mask=1 indicates that a pixel does not ground truth labels
            if 'sparse' in args.train_mode:
                depth_loss = depth_criterion(pred, batch_data['d'])
                mask = (batch_data['d'] < 1e-3).float()
            elif 'dense' in args.train_mode:
                depth_loss = depth_criterion(pred, gt)
                mask = (gt < 1e-3).float()

            # Loss 2: the self-supervised photometric loss
            if args.use_pose:
                # create multi-scale pyramids
                pred_array = helper.multiscale(pred)
                rgb_curr_array = helper.multiscale(batch_data['rgb'])
                rgb_near_array = helper.multiscale(batch_data['rgb_near'])
                if mask is not None:
                    mask_array = helper.multiscale(mask)
                num_scales = len(pred_array)

                # compute photometric loss at multiple scales
                for scale in range(len(pred_array)):
                    pred_ = pred_array[scale]
                    rgb_curr_ = rgb_curr_array[scale]
                    rgb_near_ = rgb_near_array[scale]
                    mask_ = None
                    if mask is not None:
                        mask_ = mask_array[scale]

                    # compute the corresponding intrinsic parameters
                    height_, width_ = pred_.size(2), pred_.size(3)
                    intrinsics_ = kitti_intrinsics.scale(height_, width_)

                    # inverse warp from a nearby frame to the current frame
                    warped_ = homography_from(rgb_near_, pred_,
                                              batch_data['r_mat'],
                                              batch_data['t_vec'], intrinsics_)
                    photometric_loss += photometric_criterion(
                        rgb_curr_, warped_, mask_) * (2**(scale - num_scales))

            # Loss 3: the depth smoothness loss
            smooth_loss = smoothness_criterion(pred) if args.w2 > 0 else 0

            # backprop
            loss = depth_loss + args.w1 * photometric_loss + args.w2 * smooth_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        gpu_time = time.time() - start

        # measure accuracy and record loss
        with torch.no_grad():
            mini_batch_size = next(iter(batch_data.values())).size(0)
            result = Result()
            if mode != 'test_prediction' and mode != 'test_completion':
                result.evaluate(pred.data, gt.data, photometric_loss)
            [
                m.update(result, gpu_time, data_time, mini_batch_size)
                for m in meters
            ]
            logger.conditional_print(mode, i, epoch, lr, len(loader),
                                     block_average_meter, average_meter)
            logger.conditional_save_img_comparison(mode, i, batch_data, pred,
                                                   epoch)
            logger.conditional_save_pred(mode, i, pred, epoch)

        every=100
        if i % 100 ==0: #every 100 batches/images (before it was after the entire dataset - two tabs/on if statement)
            avg = logger.conditional_save_info(mode, average_meter, epoch)
            is_best = logger.rank_conditional_save_best(mode, avg, epoch)
            if is_best and not (mode == "train"):
                logger.save_img_comparison_as_best(mode, epoch)
            logger.conditional_summarize(mode, avg, is_best)

            helper.save_checkpoint({  # save checkpoint
                'epoch': epoch,
                'model': model.module.state_dict(),
                'best_result': logger.best_result,
                'optimizer': optimizer.state_dict(),
                'args': args,
            }, is_best, epoch, logger.output_directory, args.type_feature, i, every)

    return avg, is_best


def main():
    global args
    checkpoint = None
    is_eval = False
    if args.evaluate:
        args_new = args
        if os.path.isfile(args.evaluate):
            print("=> loading checkpoint '{}' ... ".format(args.evaluate),
                  end='')
            checkpoint = torch.load(args.evaluate, map_location=device)
            args = checkpoint['args']
            args.data_folder = args_new.data_folder
            args.val = args_new.val
            is_eval = True
            print("Completed.")
        else:
            print("No model found at '{}'".format(args.evaluate))
            return
    elif args.resume:  # optionally resume from a checkpoint
        args_new = args
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}' ... ".format(args.resume),
                  end='')
            checkpoint = torch.load(args.resume, map_location=device)
            args.start_epoch = checkpoint['epoch'] + 1
            args.data_folder = args_new.data_folder
            args.val = args_new.val
            print("Completed. Resuming from epoch {}.".format(
                checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'".format(args.resume))
            return

    print("=> creating model and optimizer ... ", end='')
    model = DepthCompletionNet(args).to(device)
    model_named_params = [
        p for _, p in model.named_parameters() if p.requires_grad
    ]
    optimizer = torch.optim.Adam(model_named_params,
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    print("completed.")
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> checkpoint state loaded.")

    model = torch.nn.DataParallel(model)

    # Data loading code
    print("=> creating data loaders ... ")
    if not is_eval:
        train_dataset = KittiDepth('train', args)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=args.workers,
                                                   pin_memory=True,
                                                   sampler=None)
        print("\t==> train_loader size:{}".format(len(train_loader)))
    val_dataset = KittiDepth('val', args)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True)  # set batch size to be 1 for validation
    print("\t==> val_loader size:{}".format(len(val_loader)))

    # create backups and results folder
    logger = helper.logger(args)
    if checkpoint is not None:
        logger.best_result = checkpoint['best_result']
    print("=> logger created.")

    if is_eval:
        print("=> starting model evaluation ...")
        result, is_best = iterate("val", args, val_loader, model, None, logger,
                                  checkpoint['epoch'])
        return

    # main loop
    print("=> starting main loop ...")
    for epoch in range(args.start_epoch, args.epochs):
        print("=> starting training epoch {} ..".format(epoch))
        iterate("train", args, train_loader, model, optimizer, logger,
                epoch)  # train for one epoch
        result, is_best = iterate("val", args, val_loader, model, None, logger,
                                  epoch)  # evaluate on validation set
        helper.save_checkpoint({ # save checkpoint
            'epoch': epoch,
            'model': model.module.state_dict(),
            'best_result': logger.best_result,
            'optimizer' : optimizer.state_dict(),
            'args' : args,
        }, is_best, epoch, logger.output_directory, args.type_feature)


if __name__ == '__main__':
    main()
