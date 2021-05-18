#--data-folder /home/kamil/Dropbox/Current_research/data/kitti --resume /home/kamil/Dropbox/Current_research/depth_completion_opt/results/good/mode=dense.input=gd.resnet34.criterion=l2.lr=1e-05.bs=1.wd=0.pretrained=False.jitter=0.1.time=2021-04-01@19-36/checkpoint--1_i_16600_typefeature_None.pth.tar


#--data-folder /home/kamil/Dropbox/Current_research/data/kitti --resume /home/kamil/Dropbox/Current_research/depth_completion_opt/results/mode=dense.input=gd.resnet34.criterion=l2.lr=1e-05.bs=1.wd=0.pretrained=False.jitter=0.1.time=2021-04-26@14-59/checkpoint_qnet-0_i_1200_typefeature_sq.pth.tar
import argparse
import os
import time

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.nn.functional as F
from PIL import Image, ImageDraw


#from dataloaders.kitti_loader import load_calib, oheight, owidth, input_options, KittiDepth
from dataloaders.kitti_loader_curr import load_calib, oheight, owidth, input_options, KittiDepth
from model import DepthCompletionNetQ, DepthCompletionNetQSquare, DepthCompletionNetQSquareNet
from metrics import AverageMeter, Result
import criteria
import helper
from inverse_warp import Intrinsics, homography_from

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
                    default='gd', #if rgb then use rgb, if gd not rgb then
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
parser.add_argument('--type_feature', default="sq", choices=["sq", "lines", "None"])
parser.add_argument('--sparse_depth_source', default='nonbin')
parser.add_argument('--instancewise', default=1)


args = parser.parse_args()
args.use_pose = ("photo" in args.train_mode)
# args.pretrained = not args.no_pretrained
args.result = os.path.join('..', 'results')
args.use_rgb = ('rgb' in args.input) or args.use_pose
args.use_d = 'd' in args.input
args.use_g = 'g' in args.input
if args.use_pose:
    args.w1, args.w2 = 0.1, 0.1
else:
    args.w1, args.w2 = 0, 0
print(args)

cuda = torch.cuda.is_available() and not args.cpu
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

    torch.set_printoptions(profile="full")
    for i, batch_data in enumerate(loader):

        print("i: ", i)
        # each batch data is 1 and has three keys d, gt, g and dim [1, 352, 1216]
        start = time.time()
        batch_data = {
            key: val.to(device)
            for key, val in batch_data.items() if val is not None
        }
        gt = batch_data[
            'gt'] if mode != 'test_prediction' and mode != 'test_completion' else None

        # if args.type_feature=="sq":
        #     depth_adjustment(gt, False)

        data_time = time.time() - start

        start = time.time()
        if mode == "train":
            pred = model(batch_data)
        else:
            with torch.no_grad():
                pred = model(batch_data)

        vis=False
        if vis:
            im = batch_data['gt'].detach().cpu().numpy()
            im_sq = im.squeeze()
            plt.figure()
            plt.imshow(im_sq)
            plt.show()
            # for i in range(im_sq.shape[0]):
            #     print(f"{i} - {np.sum(im_sq[i])}")

        depth_loss, photometric_loss, smooth_loss, mask = 0, 0, 0, None
        if mode == 'train':
            # Loss 1: the direct depth supervision from ground truth label
            # mask=1 indicates that a pixel does not ground truth labels
            if 'sparse' in args.train_mode:
                depth_loss = depth_criterion(pred, batch_data['d'])
                print("d pts: ", len(torch.where(batch_data['d']>0)[0]))
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

        # counting pixels in each bin
        #binned_pixels = np.load("value.npy", allow_pickle=True)
        #print(len(binned_pixels))

        if (i % 1 == 0 and args.evaluate and args.instancewise) or (i % 5 == 0 and not args.evaluate and not args.instancewise):
            #    print(model.module.conv4[5].conv1.weight[0])
            # print(model.conv4.5.bn2.weight)
            # print(model.module.parameter.grad)
            #print("*************swiches:")
            torch.set_printoptions(precision=6, sci_mode=False)

            if model.module.phi is not None:
                mmp = 1000 * model.module.phi
                phi = F.softplus(mmp)

                S = phi / torch.sum(phi)
                print("S", S[1, -10:])
                S_numpy= S.detach().cpu().numpy()

            if args.instancewise:

                global Ss
                if "Ss" not in globals():
                    Ss = []
                    Ss.append(S_numpy)
                else:
                    Ss.append(S_numpy)


            if (i % 5 ==0  and not args.evaluate and not args.instancewise and model.module.phi is not None):

                switches_2d_argsort = np.argsort(S_numpy, None) # 2d to 1d sort torch.Size([9, 31])
                switches_2d_sort = np.sort(S_numpy, None)
                print("Switches: ")
                print(switches_2d_argsort[:10])
                print(switches_2d_sort[:10])
                print("and")
                print(switches_2d_argsort[-10:])
                print(switches_2d_sort[-10:])

                if args.type_feature == "sq":

                    hor = switches_2d_argsort % S_numpy.shape[1]
                    ver = np.floor(switches_2d_argsort // S_numpy.shape[1])
                    print(ver[:10],hor[:10])
                    print("and")
                    print(ver[-10:], hor[-10:])
                    # if type_feature == "sq":
                    #     print(switches_2d_argsort)
                    # elif type_feature == "lines":
                    #     print(torch.argsort(S))
                    np.save(f"ranks/switches_argsort_2D_equal_iter_{i}.npy", switches_2d_argsort)
                    np.save(f"ranks/switches_2D_equal_iter_{i}.npy", S_numpy)

                elif args.type_feature == "lines":
                    np.save(f"ranks/switches_argsort_2D_equal_lines_iter_{i}.npy", switches_2d_argsort)
                    np.save(f"ranks/switches_2D_equal_lines_iter_{i}.npy", S_numpy)
                    if "last_argsort" in locals():
                        os.remove(last_argsort)
                        os.remove(last_switches)
                    last_argsort = f"ranks/switches_argsort_2D_equal_lines_iter_{i}.npy"
                    last_switches =  f"ranks/switches_2D_equal_lines_iter_{i}.npy"



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

        draw=False
        if draw:
            ma = batch_data['rgb'].detach().cpu().numpy().squeeze()
            ma  = np.transpose(ma, axes=[1, 2, 0])
           # ma = np.uint8(ma)
            #ma2 = Image.fromarray(ma)
            ma2 = Image.fromarray(np.uint8(ma)).convert('RGB')
            # create rectangle image
            img1 = ImageDraw.Draw(ma2)

            if args.type_feature == "sq":
                size=40
                print_square_num = 20
                for ii in range(print_square_num):
                    s_hor=hor[-ii].detach().cpu().numpy()
                    s_ver=ver[-ii].detach().cpu().numpy()
                    #print("Top square switches: ")
                    #print(s_ver, s_hor)
                    shape = [(s_hor * size, s_ver * size), ((s_hor + 1) * size, (s_ver + 1) * size)]
                    #print("shape: ", shape)

                    img1.rectangle(shape, outline="red")

                    tim = time.time()
                    lala = ma2.save(f"switches_photos/squares/squares_{tim}.jpg")
                    print("saving")
            elif args.type_feature == "lines":
                print_square_num = 20
                r=1
                parameter_mask = np.load("../kitti_pixels_to_lines.npy", allow_pickle=True)

                # for m in range(10,50):
                #     im = Image.fromarray(parameter_mask[m]*155)
                #     im = im.convert('1')  # convert image to black and white
                #     im.save(f"switches_photos/lala_{m}.jpg")


                for ii in range(print_square_num):
                     points = parameter_mask[ii]
                     y = np.where(points==1)[0]
                     x = np.where(points == 1)[1]

                     for p in range(len(x)):
                         img1.ellipse((x[p] - r, y[p] - r, x[p] + r, y[p] + r), fill=(255, 0, 0, 0))

                tim = time.time()
                lala = ma2.save(f"switches_photos/lines/lines_{tim}.jpg")
                print("saving")





        every = 50
        if i % every ==0:

            print("saving")
            avg = logger.conditional_save_info(mode, average_meter, epoch)
            is_best = logger.rank_conditional_save_best(mode, avg, epoch)
            if is_best and not (mode == "train"):
                logger.save_img_comparison_as_best(mode, epoch)
            logger.conditional_summarize(mode, avg, is_best)

            if mode != "val":
                helper.save_checkpoint({  # save checkpoint
                    'epoch': epoch,
                    'model': model.module.state_dict(),
                    'best_result': logger.best_result,
                    'optimizer': optimizer.state_dict(),
                    'args': args,
                }, is_best, epoch, logger.output_directory, args.type_feature, i, every)

    if args.evaluate:
        filename = os.path.split(args.evaluate)[1]
        Ss_numpy = np.array(Ss)
        np.save(f"ranks/instance/Ss_val_{filename}.npy", Ss)

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
            args.evaluate = args_new.evaluate
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
            args.sparse_depth_source = args_new.sparse_depth_source
            args.val = args_new.val
            print("Completed. Resuming from epoch {}.".format(
                checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'".format(args.resume))
            return

    print("=> creating model and optimizer ... ", end='')

    # model
    if args.type_feature == "sq":
        if args.instancewise:
            model = DepthCompletionNetQSquareNet(args).to(device)
        else:
            model = DepthCompletionNetQSquare(args).to(device)
    elif args.type_feature == "lines":
        model = DepthCompletionNetQ(args).to(device)
    model_named_params = [
        p for _, p in model.named_parameters() if p.requires_grad
    ]
    optimizer = torch.optim.Adam(model_named_params,
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    print("completed.")
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'], strict=False)
        #optimizer.load_state_dict(checkpoint['optimizer'])
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

    for name, param in model.named_parameters():
    #for name, param in model.state_dict().items():
        #print(name, param.shape)
        if "parameter" not in name:
        #if 1:
            h = param.register_hook(lambda grad: grad * 0)  # double the gradient

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
        }, is_best, epoch, logger.output_directory)


if __name__ == '__main__':
    main()
