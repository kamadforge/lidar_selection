# --data-folder /home/kamil/Dropbox/Current_research/data/kitti -e /home/kamil/Dropbox/Current_research/depth_completion_opt/results/good/mode=dense.input=gd.resnet34.criterion=l2.lr=1e-05.bs=1.wd=0.pretrained=False.jitter=0.1.time=2021-04-01@19-36/checkpoint--1_i_16600_typefeature_None.pth.tar

# saved depth_predicted

import argparse
import os
import time
import datetime

now = datetime.datetime.now()
date_time = now.strftime("%Y_%m_%d_%H:%M")
import sys

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data

# from dataloaders.kitti_loader_apr12 import load_calib, oheight, owidth, input_options, KittiDepth
from dataloaders.kitti_loader_curr_jul26 import load_calib, oheight, owidth, input_options, KittiDepth
from model import DepthCompletionNet
from metrics import AverageMeter, Result
import criteria
import helper
from inverse_warp import Intrinsics, homography_from
from features.depth_manipulation import depth_adjustment, depth_adjustment_lines
#from features.depth_manipulation_oct31_localvaishakh_remove import depth_adjustment_lines
from features.depth_draw import draw
import vis_utils

import numpy as np

# ARGUMENTS

parser = argparse.ArgumentParser(description='Sparse-to-Dense')
parser.add_argument('-w', '--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=11, type=int, metavar='N', help='number of total epochs to run (default: 11)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-c', '--criterion', metavar='LOSS', default='l2', choices=criteria.loss_names, help='loss function: | '.join(criteria.loss_names) + ' (default: l2)')
parser.add_argument('-b', '--batch-size', default=1, type=int, help='mini-batch size (default: 1)')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float, metavar='LR',  help='initial learning rate (default 1e-5)')
parser.add_argument('--weight-decay', '--wd', default=0, type=float, metavar='W', help='weight decay (default: 0)')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--data-folder', default='../data', type=str, metavar='PATH', help='data folder (default: none)')
parser.add_argument('-i', '--input', type=str, default='gd', choices=input_options,help='input: | '.join(input_options))
parser.add_argument('-l', '--layers', type=int, default=18, help='use 16 for sparse_conv; use 18 or 34 for resnet')
parser.add_argument('--pretrained', action="store_true", help='use ImageNet pre-trained weights')
parser.add_argument('--val', type=str, default="select", choices=["select", "full"], help='full or select validation set')
parser.add_argument('--jitter', type=float, default=0.1, help='color jitter for images')
parser.add_argument('--rank-metric', type=str, default='rmse', choices=[m for m in dir(Result()) if not m.startswith('_')], help='metrics for which best result is sbatch_datacted')
parser.add_argument('-m', '--train-mode', type=str, default="dense", choices=["dense", "sparse", "photo", "sparse+photo", "dense+photo"], help='dense | sparse | photo | sparse+photo | dense+photo')
parser.add_argument('-e', '--evaluate', default='3', type=str, metavar='PATH')
parser.add_argument('--record_eval_shap', default=0, type=int)
parser.add_argument('--cpu', action="store_true", help='run on cpu')
parser.add_argument('--depth_adjust', default=1, type=int)  # if we use all depth or subset of depth feature
parser.add_argument('--sparse_depth_source', default='nonbin')
parser.add_argument('--depth_save', default=1, type=int)
parser.add_argument('--seed', default=120, type=int)
parser.add_argument('--type_feature', default="lines", choices=["sq", "lines", "None"])
parser.add_argument('--test_mode', default="all")
parser.add_argument('--feature_mode', default='local')
parser.add_argument('--feature_num', default=32, type=int)
parser.add_argument('--ranks_file', default="/home/kamil/Dropbox/Current_research/depth_completion_opt/self-supervised-depth-completion-master2_working/ranks/lines/global/16600_switches_2D_equal_iter_3990.npy")
# "/home/kamil/Dropbox/Current_research/depth_completion_opt/self-supervised-depth-completion-master2_working/ranks/lines/global/checkpoint_10_i_85000__best.pth.tar/global/Ss_val_ep_11_it_7.npy")
# "/home/kamil/Dropbox/Current_research/depth_completion_opt/self-supervised-depth-completion-master2_working/ranks/lines/global/16600_switches_2D_equal_iter_3990.npy"
parser.add_argument('--rank_file_global_sq')
parser.add_argument('--draw_features_rgb', default=0)
args = parser.parse_args()
if args.evaluate == "1":
    args.evaluate = "/home/kamil/Dropbox/Current_research/depth_completion_opt/results/good/mode=dense.input=gd.resnet34.criterion=l2.lr=1e-05.bs=1.wd=0.pretrained=False.jitter=0.1.time=2021-04-01@19-36/checkpoint--1_i_16600_typefeature_None.pth.tar"
elif args.evaluate == "2":
    args.evaluate = "/home/kamil/Dropbox/Current_research/depth_completion_opt/results/good/mode=dense.input=gd.resnet34.criterion=l2.lr=1e-05.bs=1.wd=0.pretrained=False.jitter=0.1.time=2021-05-24@22-50_2/checkpoint_qnet-9_i_0_typefeature_None.pth.tar"
elif args.evaluate =="3":
    args.evaluate = "/home/kamil/Dropbox/Current_research/depth_completion_opt/results/good/checkpoint_10_i_85000__best.pth.tar"
args.use_pose = ("photo" in args.train_mode)
# args.pretrained = not args.no_pretrained
# args.result = os.path.join('..', 'results')
# args.use_rgb = ('rgb' in args.input) or args.use_pose
args.use_rgb = True
args.use_d = 'd' in args.input
args.use_g = 'g' in args.input
if args.use_pose:
    args.w1, args.w2 = 0.1, 0.1
else:
    args.w1, args.w2 = 0, 0
model_orig = os.path.split(args.evaluate)[1]
if args.evaluate != "":
    args.result = f"../results/val/{date_time}"
else:
    args.result = f"../results/train/{date_time}"
evaluate_path_split=args.evaluate.split("/")
if args.record_eval_shap:
    os.makedirs(f"ranks/lines/instance/shap/" + evaluate_path_split[-1], exist_ok=True)
os.makedirs(args.result, exist_ok=True)
args.seed = int(time.time()) + np.random.choice(100000)
print(args)
print(sys.argv[1:])

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


#########################################################################################

def iterate(mode, args, loader, model, optimizer, logger, epoch):
    block_average_meter = AverageMeter()
    average_meter = AverageMeter()
    meters = [block_average_meter, average_meter]
    # switch to appropriate mode
    assert mode in ["train", "val", "eval", "test_prediction", "test_completion"], "unsupported mode: {}".format(mode)
    if mode == 'train':
        model.train()
        lr = helper.adjust_learning_rate(args.lr, optimizer, epoch)
    else:
        model.eval()
        lr = 0
    torch.set_printoptions(profile="full")

    # ITERATE OVER IMAGES

    for i, batch_data in enumerate(loader):

        sparse_depth_pathname = batch_data['d_path'][0]

        print(sparse_depth_pathname)
        depth_path_split = sparse_depth_pathname.split("/")
        filename = depth_path_split[-1][:-4]
        if args.record_eval_shap:
            filepath = f"ranks/lines/instance/shap/{evaluate_path_split[-1]}/{filename}.txt"
            if not os.path.exists(filepath):
                open(filepath, "w").close()
        del batch_data['d_path']
        print("i: ", i)
        if args.use_d:
            print(f"depth (sparse) points: {len(torch.where(batch_data['d'] > 0)[0])}")
            print(f"gt depth (dense) points: {len(torch.where(batch_data['gt'] > 0)[0])}")
        start = time.time()
        batch_data = {
            key: val.to(device)
            for key, val in batch_data.items() if val is not None
        }
        gt = batch_data['gt'] if mode != 'test_prediction' and mode != 'test_completion' else None

        # adjust depth for features
        depth_adjust = args.depth_adjust
        adjust_features = False  # normalize the number of points in a feature
        if depth_adjust and args.use_d:
            if args.type_feature == "sq":
                if args.use_rgb:
                    depth_new, alg_mode, feat_mode, features, shape = depth_adjustment(batch_data['d'], args.test_mode,args.feature_mode,args.feature_num,adjust_features, i, model_orig,args.seed, batch_data['rgb'])
                else:
                    depth_new, alg_mode, feat_mode, features, shape = depth_adjustment(batch_data['d'], args.test_mode,args.feature_mode,args.feature_num,adjust_features, i, model_orig,args.seed)
            elif args.type_feature == "lines":
                depth_new, alg_mode, feat_mode, features = depth_adjustment_lines(batch_data['d'], args.test_mode,args.feature_mode, args.feature_num,i, model_orig, sparse_depth_pathname, args.seed)

            batch_data['d'] = torch.Tensor(depth_new).unsqueeze(0).unsqueeze(1).to(device)
        data_time = time.time() - start
        start = time.time()
        if mode == "train":
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

        # compute loss
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
                    warped_ = homography_from(rgb_near_, pred_, batch_data['r_mat'],batch_data['t_vec'], intrinsics_)
                    photometric_loss += photometric_criterion(
                        rgb_curr_, warped_, mask_) * (2 ** (scale - num_scales))
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
            [m.update(result, gpu_time, data_time, mini_batch_size) for m in meters]

            print(f"rmse: {result.rmse:,}")
            if result.rmse < 6000:
                print("good rmse")
            elif result.rmse > 12000:
                print("bad rmse")

            logger.conditional_print(mode, i, epoch, lr, len(loader), block_average_meter, average_meter)
            logger.conditional_save_img_comparison(mode, i, batch_data, pred, epoch)
            logger.conditional_save_pred(mode, i, pred, epoch)

            if args.record_eval_shap:
                with open(filepath, "a+") as file:
                    file.write("\n" + ",".join([str(f) for f in features]) + ":" + "{:.3f}".format(result.rmse))

        # save log and checkpoint
        every = len(loader) - 1 if mode == "val" else 1000  # 200

        if i % every == 0 and i != 0:
            print(datetime.datetime.now())
            print(f"test settings (main_orig eval): {args.type_feature} {args.test_mode} {args.feature_mode} {args.feature_num}")
            avg = logger.conditional_save_info(mode, average_meter, epoch)
            is_best = logger.rank_conditional_save_best(mode, avg, epoch)
            if is_best and not (mode == "train"):
                logger.save_img_comparison_as_best(mode, epoch)
            logger.conditional_summarize(mode, avg, is_best, args, i)

            if mode != "val":
                # if 1:
                helper.save_checkpoint({  # save checkpoint
                    'epoch': epoch,
                    'model': model.module.state_dict(),
                    'best_result': logger.best_result,
                    'optimizer': optimizer.state_dict(),
                    'args': args,}, is_best, epoch, logger.output_directory, args.type_feature, args.test_mode, args.feature_num, args.feature_mode, args.depth_adjust, i, every, "scratch")

        # draw features
        if args.draw_features_rgb and args.evaluate and depth_adjust:
            run_info = [args.type_feature, alg_mode, feat_mode, model_orig]
            if batch_data['rgb'] != None and 1 and (i % 1) == 0:
                draw(args.type_feature, batch_data['rgb'], batch_data['d'], features, 65, run_info, i, result)
        if args.depth_save and args.evaluate:
            name = os.path.split(sparse_depth_pathname)[-1]
            parameters_name = args.evaluate.split(os.sep)
            if depth_adjust:
                path_depth = f"depth_predicted/{args.type_feature}/{args.feature_mode}/{args.test_mode}/{parameters_name[-2]}/{parameters_name[-1]}/"
            else:
                path_depth = f"depth_predicted/full/{parameters_name[-2]}/{parameters_name[-1]}/"
            os.makedirs(path_depth, exist_ok=True)
            # torch.save(pred, path_depth + f"{name}.pt")
            depth_predicted = pred.squeeze().detach().cpu().numpy()
            depth_pred_color = vis_utils.depth_colorize(depth_predicted)
            vis_utils.save_depth_as_uint16png(depth_pred_color, path_depth + f"{name}_im_color.png")

        print('\n' + '*' * 15 + '\n\n')

    if args.record_eval_shap:
        with open("ranks/lines/global/shap/lines_shap.txt", "a+") as file:
            file.write("\n" + ",".join([str(f) for f in features]) + ":" + "{:.3f}".format(avg.rmse))

    return avg, is_best


############################################################################

def main():
    global args
    checkpoint = None
    is_eval = False
    if args.evaluate:
        if os.path.isfile(args.evaluate):
            print("=> loading checkpoint '{}' ... ".format(args.evaluate), end='')
            checkpoint = torch.load(args.evaluate, map_location=device)
            is_eval = True
            print("Completed.")
        else:
            print("No model found at '{}'".format(args.evaluate))
            return
    elif args.resume:  # optionally resume from a checkpoint
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}' ... ".format(args.resume), end='')
            checkpoint = torch.load(args.resume, map_location=device)
            args.start_epoch = checkpoint['epoch'] + 1
            print("Completed. Resuming from epoch {}.".format(checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'".format(args.resume))
            return

    print("=> creating model and optimizer ... ", end='')
    model = DepthCompletionNet(args).to(device)
    model_named_params = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(model_named_params, lr=args.lr, weight_decay=args.weight_decay)
    print("completed.")
    # pretrained
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> checkpoint state loaded.")
    model = torch.nn.DataParallel(model)

    # DATA
    print("=> creating data loaders ... ")
    if not is_eval:
        # train dataset
        train_dataset = KittiDepth('train', args)
        train_dataset_sub = torch.utils.data.Subset(train_dataset, torch.arange(10))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, sampler=None)
        print("\t==> train_loader size:{}".format(len(train_loader)))
    # test dataset
    val_dataset = KittiDepth('val', args)
    val_dataset_sub = torch.utils.data.Subset(val_dataset, torch.arange(1))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0,pin_memory=True)  # set batch size to be 1 for validation
    print("\t==> val_loader size:{}".format(len(val_loader)))

    # create backups and results folder
    logger = helper.logger(args)
    if checkpoint is not None:
        logger.best_result = checkpoint['best_result']
    print("=> logger created.")

    if is_eval:
        print("\n\n\n=> starting model evaluation ...\n\n\n\n")
        result, is_best = iterate("val", args, val_loader, model, None, logger, checkpoint['epoch'])
        print("Res:", result)
        print(is_best)
        return

    # MAIN LOOP
    print("=> starting main loop ...")
    for epoch in range(args.start_epoch, args.epochs):
        print("=> starting training epoch {} ..".format(epoch))
        # train for one epoch
        iterate("train", args, train_loader, model, optimizer, logger, epoch)
        # evaluate on the val set
        result, is_best = iterate("val", args, val_loader, model, None, logger, epoch)
        # save
        helper.save_checkpoint({'epoch': epoch, 'model': model.module.state_dict(), 'best_result': logger.best_result,'optimizer': optimizer.state_dict(), 'args': args}, is_best, epoch, logger.output_directory, args.type_feature, args.test_mode, args.feature_num,args.feature_mode, args.depth_adjust)


if __name__ == '__main__':
    t = time.time()
    main()
    print("Total time: ", (t - time.time()) / 60.)
