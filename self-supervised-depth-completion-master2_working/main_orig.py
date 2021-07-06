#--data-folder /home/kamil/Dropbox/Current_research/data/kitti -e /home/kamil/Dropbox/Current_research/depth_completion_opt/results/good/mode=dense.input=gd.resnet34.criterion=l2.lr=1e-05.bs=1.wd=0.pretrained=False.jitter=0.1.time=2021-04-01@19-36/checkpoint--1_i_16600_typefeature_None.pth.tar

#saved depth_predicted

import argparse
import os
import time

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data

from dataloaders.kitti_loader_apr12 import load_calib, oheight, owidth, input_options, KittiDepth
from model import DepthCompletionNet
from metrics import AverageMeter, Result
import criteria
import helper
from inverse_warp import Intrinsics, homography_from
from features.depth_manipulation import depth_adjustment, depth_adjustment_lines
from features.depth_draw import draw

import numpy as np

# ARGUMENTS

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
                    default=1e-5, #1e-5
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
                    default='gd', #'gd' greyscale and depth
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
# parser.add_argument('-e', '--evaluate', default='/home/kamil/Dropbox/Current_research/depth_completion_opt/results/good/mode=dense.input=gd.resnet34.criterion=l2.lr=1e-05.bs=1.wd=0.pretrained=False.jitter=0.1.time=2021-04-01@19-36/checkpoint--1_i_16600_typefeature_None.pth.tar')

# parser.add_argument('-e', '--evaluate', default="/home/kamil/Dropbox/Current_research/depth_completion_opt/results/good/mode=dense.input=d.resnet34.criterion=l2.lr=1e-05.bs=1.wd=0.pretrained=False.jitter=0.1.time=2021-05-03@21-17/checkpoint--1_i_85850_typefeature_None.pth.tar")

parser.add_argument('--cpu', action="store_true", help='run on cpu')
parser.add_argument('--type_feature', default="sq", choices=["sq", "lines", "None"])
parser.add_argument('--depth_adjust', default=1, type=int)
parser.add_argument('--sparse_depth_source', default='nonbin')
#parser.add_argument('--ranks_file', nargs="+", default=["la", "la"])
parser.add_argument('--ranks_file', default="/home/kamil/Dropbox/Current_research/depth_completion_opt/self-supervised-depth-completion-master2_working/ranks/lines/global/16600_switches_2D_equal_iter_3990.npy")
args = parser.parse_args()

if args.evaluate == "1":
    args.evaluate = "/home/kamil/Dropbox/Current_research/depth_completion_opt/results/good/mode=dense.input=gd.resnet34.criterion=l2.lr=1e-05.bs=1.wd=0.pretrained=False.jitter=0.1.time=2021-04-01@19-36/checkpoint--1_i_16600_typefeature_None.pth.tar"
elif args.evaluate == "2":
    args.evaluate = "/home/kamil/Dropbox/Current_research/depth_completion_opt/results/good/mode=dense.input=gd.resnet34.criterion=l2.lr=1e-05.bs=1.wd=0.pretrained=False.jitter=0.1.time=2021-05-24@22-50_2/checkpoint_qnet-9_i_0_typefeature_None.pth.tar"

args.use_pose = ("photo" in args.train_mode)
# args.pretrained = not args.no_pretrained
args.result = os.path.join('..', 'results')
#args.use_rgb = ('rgb' in args.input) or args.use_pose
args.use_rgb = True
args.use_d = 'd' in args.input
args.use_g = 'g' in args.input
if args.use_pose:
    args.w1, args.w2 = 0.1, 0.1
else:
    args.w1, args.w2 = 0, 0

if args.evaluate:
    args.ranks_file = args.ranks_file.split(os.sep)[-3:] #folder and name

print(args)




print("\nEvaluate: ", args.evaluate)
print(f"input: {args.input} from {args.sparse_depth_source}")
if args.depth_adjust:
    print("with depth adjust in main_orig to squares")
else:
    print("vanilla full depth")
print(f"\noutput (only for training): {args.train_mode}\n")


# SET-UP

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


def iterate_eval_simple(mode, args, loader, model, optimizer, logger, epoch):
    block_average_meter = AverageMeter()
    average_meter = AverageMeter()
    meters = [block_average_meter, average_meter]

    # switch to appropriate mode
    assert mode in ["val", "eval", "test_prediction", "test_completion"], \
        "unsupported mode: {}".format(mode)

    model.eval()
    lr = 0

    torch.set_printoptions(profile="full")
    for i, batch_data in enumerate(loader):

        print ("i: ", i)
        start = time.time()
        batch_data = {
            key: val.to(device)
            for key, val in batch_data.items() if val is not None and type(val) != str
        }
        gt = batch_data[
            'gt'] if mode != 'test_prediction' and mode != 'test_completion' else None

        original_depth = batch_data['d'].data
        # a test to run the photo several times on different subsets of features

        #if i in [83, 260, 324, 413, 150, 295, 303, 310, 466]:
        if i in [413, 466]:
            it_nums = 500
        else:
            it_nums = 0

        best = 10000000000000000
        worst = 0
        for it_random_test in range(it_nums):
            print("it_random_test: ", it_random_test)

            batch_data['d'] = original_depth.data


            # adjust depth
            depth_adjust=args.depth_adjust
            adjust_features=False
            if depth_adjust and args.use_d:
                if args.type_feature == "sq":
                    if args.use_rgb:
                        depth_new = depth_adjustment(batch_data['d'], adjust_features, i, args.ranks_file, batch_data['rgb'],it_random_test, )
                    else:
                        depth_new = depth_adjustment(batch_data['d'], adjust_features, i,  args.ranks_file,None, it_random_test)
                elif args.type_feature == "lines":
                    depth_new = depth_adjustment_lines(batch_data['d'])


                batch_data['d'] = torch.Tensor(depth_new).unsqueeze(0).unsqueeze(1).to(device)
            data_time = time.time() - start
            start = time.time()
            if mode=="train":
                pred = model(batch_data)
            else:
                with torch.no_grad():
                    pred = model(batch_data)
            # im = batch_data['d'].detach().cpu().numpy()
            # im_sq = im.squeeze()
            # plt.figure()
            # plt.imshow(im_sq)
            # plt.show()
            # for i in range(im_sq.shape[0]):
            #     print(f"{i} - {np.sum(im_sq[i])}")


            # computing mse error only
            output = pred.data
            target = gt.data
            valid_mask = target > 0.1
            print("output", output.sum())
            # convert from meters to mm
            output_mm = 1e3 * output[valid_mask]
            target_mm = 1e3 * target[valid_mask]
            abs_diff = (output_mm - target_mm).abs()
            print("output_mm_1", output_mm.sum())
            mse = float((torch.pow(abs_diff, 2)).mean())
            print("mse::", mse)
            if mse>worst:
                worst = mse; worst_id = (i, it_random_test)
            if mse<best:
                best = mse; best_id = (i, it_random_test)

        if worst !=0 :
            print(f"Best: {best_id}, {best}")
            print(f"Worse: {worst_id}, {worst}")



        # every=990 if mode == "val" else 200
        # if i % every ==0:
        #
        #     print("saving")
        #     avg = logger.conditional_save_info(mode, average_meter, epoch)
        #     is_best = logger.rank_conditional_save_best(mode, avg, epoch)
        #     if is_best and not (mode == "train"):
        #         logger.save_img_comparison_as_best(mode, epoch)
        #     logger.conditional_summarize(mode, avg, is_best)
        #
        #     if mode != "val":
        #     #if 1:
        #         helper.save_checkpoint({  # save checkpoint
        #             'epoch': epoch,
        #             'model': model.module.state_dict(),
        #             'best_result': logger.best_result,
        #             'optimizer': optimizer.state_dict(),
        #             'args': args,
        #         }, is_best, epoch, logger.output_directory, args.type_feature, i, every)

    return avg, is_best

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
    table_is=np.zeros(400)

    # ITERATE OVER IMAGES

    for i, batch_data in enumerate(loader):

        # name = batch_data['name'][0]
        # print(name)
        # del batch_data['name']
        print ("i: ", i)
        start = time.time()
        batch_data = {
            key: val.to(device)
            for key, val in batch_data.items() if val is not None
        }
        gt = batch_data[
            'gt'] if mode != 'test_prediction' and mode != 'test_completion' else None

        # adjust depth
        depth_adjust=args.depth_adjust
        adjust_features=False
        if depth_adjust and args.use_d and args.evaluate:
            if args.type_feature == "sq":
                if args.use_rgb:
                    depth_new = depth_adjustment(batch_data['d'], adjust_features, i, args.ranks_file, batch_data['rgb'])
                else:
                    depth_new = depth_adjustment(batch_data['d'], adjust_features, i, args.ranks_file)
            elif args.type_feature == "lines":
                depth_new = depth_adjustment_lines(batch_data['d'], adjust_features, i, args.ranks_file)

            batch_data['d'] = torch.Tensor(depth_new).unsqueeze(0).unsqueeze(1).to(device)
        data_time = time.time() - start
        start = time.time()
        if mode=="train":
            pred = model(batch_data)
        else:
            with torch.no_grad():
                pred = model(batch_data)
                #torch.save(pred, f"depth_predicted/{name[:-4]}.t")
        # im = batch_data['d'].detach().cpu().numpy()
        # im_sq = im.squeeze()
        # plt.figure()
        # plt.imshow(im_sq)
        # plt.show()
        # for i in range(im_sq.shape[0]):
        #     print(f"{i} - {np.sum(im_sq[i])}")


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
            print(f"rmse: {int(result.rmse)}")
            if result.rmse > 12500:
                print("bad rmse")
            elif result.rmse < 4000:
                print("good")

            logger.conditional_print(mode, i, epoch, lr, len(loader),
                                     block_average_meter, average_meter)
            logger.conditional_save_img_comparison(mode, i, batch_data, pred,
                                                   epoch)
            logger.conditional_save_pred(mode, i, pred, epoch)



        every=999 if mode == "val" else 200
        if i % every ==0:

            print("saving")
            avg = logger.conditional_save_info(mode, average_meter, epoch)
            is_best = logger.rank_conditional_save_best(mode, avg, epoch)
            if is_best and not (mode == "train"):
                logger.save_img_comparison_as_best(mode, epoch)
            logger.conditional_summarize(mode, avg, is_best)

            if mode != "val":
            #if 1:
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
            args.type_feature = args_new.type_feature
            args.data_folder = args_new.data_folder
            args.val = args_new.val
            args.sparse_depth_source = args_new.sparse_depth_source
            args.depth_adjust = args_new.depth_adjust
            args.use_rgb = args_new.use_rgb
            args.use_d = args_new.use_d
            args.input = args_new.input
            args.evaluate = args_new.evaluate
            args.ranks_file = args_new.ranks_file
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

    # DATA

    print("=> creating data loaders ... ")
    if not is_eval:
        train_dataset = KittiDepth('train', args)
        train_dataset_sub = torch.utils.data.Subset(train_dataset, torch.arange(10))
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=args.workers,
                                                   pin_memory=True,
                                                   sampler=None)
        print("\t==> train_loader size:{}".format(len(train_loader)))
    val_dataset = KittiDepth('val', args)

    val_dataset_sub = torch.utils.data.Subset(val_dataset, torch.arange(1000))
    val_loader = torch.utils.data.DataLoader(
        val_dataset_sub,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)  # set batch size to be 1 for validation
    print("\t==> val_loader size:{}".format(len(val_loader)))

    # create backups and results folder
    logger = helper.logger(args)
    if checkpoint is not None:
        logger.best_result = checkpoint['best_result']
    print("=> logger created.")

    if is_eval:
        print("=> starting model evaluation ...")
        result, is_best = iterate("val", args, val_loader, model, None, logger, checkpoint['epoch'])
        return

    # MAIN LOOP

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
    t=time.time()
    main()
    print("Total time: ", (t-time.time())/60.)
