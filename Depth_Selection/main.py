import argparse
import numpy as np
import os
import sys
import time
import shutil
import glob
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim
import Models
import Datasets
from tensorboardX import SummaryWriter
import json
from torchvision.utils import make_grid
import warnings
import random
from datetime import datetime
from Loss.loss import define_loss, allowed_losses, MSE_loss
from Loss.benchmark_metrics import Metrics, allowed_metrics
from Datasets.dataloader import get_loader
import pdb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Utils.utils import str2bool, define_optim, define_scheduler, \
                        Logger, AverageMeter, first_run, mkdir_if_missing, \
                        define_init_weights, init_distributed_mode, merge_into_row, merge_into_row_with_gt, \
                        save_image, add_row

# Training setttings
parser = argparse.ArgumentParser(description='KITTI Depth Completion Task')
parser.add_argument('--dataset', type=str, default='kitti', choices=Datasets.allowed_datasets(), help='dataset to work with')
parser.add_argument('--nepochs', type=int, default=100, help='Number of epochs for training')
parser.add_argument('--thres', type=int, default=0, help='epoch for pretraining')
parser.add_argument('--start_epoch', type=int, default=0, help='Start epoch number for training')
parser.add_argument('--mod', type=str, default='mod', choices=Models.allowed_models(), help='Model for use')
parser.add_argument('--submod', type=str, default='resnet', help='model in refinenet')
parser.add_argument('--batch_size', type=int, default=7, help='batch size')
parser.add_argument('--layers', type=int, default=34, help='batch size')
parser.add_argument('--val_batch_size', default=None, help='batch size selection validation set')
parser.add_argument('--learning_rate', metavar='lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--drop', type=float, default=0.3, help='dropout')
parser.add_argument('--no_cuda', action='store_true', help='no gpu usage')
parser.add_argument('--modality', type=str, default='rgbd', help='input modality(rgb, rgbd)')

parser.add_argument('--evaluate', action='store_true', help='only evaluate')
parser.add_argument('--resume', type=str, default='', help='resume latest saved run')
parser.add_argument('--nworkers', type=int, default=8, help='num of threads')
parser.add_argument('--nworkers_val', type=int, default=0, help='num of threads')
parser.add_argument('--no_dropout', action='store_true', help='no dropout in network')
parser.add_argument('--subset', type=int, default=None, help='Take subset of train set')
parser.add_argument('--input_type', type=str, default='rgb', choices=['depth','rgb'], help='use rgb for rgbdepth')
parser.add_argument('--no_tb', type=str2bool, nargs='?', const=True,
                    default=True, help="use mask_gt - mask_input as final mask for loss calculation")
parser.add_argument('--test_mode', action='store_true', help='Do not use resume')
parser.add_argument('--pretrained', type=str2bool, nargs='?', const=True, default=True, help='use pretrained model')
parser.add_argument('--load_external_mod', type=str2bool, nargs='?', const=True, default=False, help='path to external mod')
parser.add_argument('--activation_lstm', type=str, default='elu', choices=['relu', 'tanh', 'none', 'elu'], help='activation function to use for the state in the lstm')
parser.add_argument('--train_init_state', action='store_true', help='no gpu usage')

# Data augmentation settings
parser.add_argument('--crop_w', type=int, default=1216, help='width of image after cropping')
parser.add_argument('--crop_h', type=int, default=256, help='height of image after cropping')
parser.add_argument('--max_depth', type=float, default=85.0, help='maximum depth of LIDAR input')
parser.add_argument('--sparse_val', type=float, default=0.0, help='value to endode sparsity with')
parser.add_argument("--rotate", type=str2bool, nargs='?', const=True, default=False, help="rotate image")
parser.add_argument("--flip", type=str, default='hflip', help="flip image: vertical|horizontal")
parser.add_argument("--rescale", type=str2bool, nargs='?', const=True,
                    default=False, help="Rescale values of sparse depth input randomly")
parser.add_argument("--normal", type=str2bool, nargs='?', const=True, default=False, help="normalize depth/rgb input")
parser.add_argument("--no_aug", type=str2bool, nargs='?', const=True, default=False, help="rotate image")

# Paths settings
parser.add_argument('--save_path', default='Saved/', help='save path')
parser.add_argument('--data_path', default='/usr/data/tmp/Depth_Completion/data', help='path to desired dataset')

# Optimizer settings
parser.add_argument('--optimizer', type=str, default='adam', help='adam or sgd')
parser.add_argument('--weight_init', type=str, default='kaiming', help='normal, xavier, kaiming, orhtogonal weights initialisation')
parser.add_argument('--weight_decay', type=float, default=0, help='L2 weight decay/regularisation on?')
parser.add_argument('--lr_decay', action='store_true', help='decay learning rate with rule')
parser.add_argument('--niter', type=int, default=50, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=400, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr_policy', type=str, default=None, help='{}learning rate policy: lambda|step|plateau')
parser.add_argument('--lr_decay_iters', type=int, default=5, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--clip_grad_norm', type=int, default=0, help='performs gradient clipping')
parser.add_argument('--gamma', type=float, default=0.5, help='factor to decay learning rate every lr_decay_iters with')

# Loss settings
parser.add_argument('--loss_criterion', type=str, default='mse', choices=allowed_losses(), help="loss criterion")
parser.add_argument('--print_freq', type=int, default=10000, help="print every x iterations")
parser.add_argument('--save_freq', type=int, default=100000, help="save every x interations")
parser.add_argument('--metric', type=str, default='rmse', choices=allowed_metrics(), help="metric to use during evaluation")
parser.add_argument('--metric_1', type=str, default='mae', choices=allowed_metrics(), help="metric to use during evaluation")
parser.add_argument('--wlid', type=float, default=0.1, help="weight base loss")
parser.add_argument('--wrgb', type=float, default=0.1, help="weight base loss")
parser.add_argument('--wguide', type=float, default=0.1, help="weight base loss")
# Cudnn
parser.add_argument("--cudnn", type=str2bool, nargs='?', const=True,
                    default=True, help="cudnn optimization active")
parser.add_argument('--gpu_ids', default='1', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument("--multi", type=str2bool, nargs='?', const=True,
                    default=False, help="use multiple gpus")
parser.add_argument("--seed", type=str2bool, nargs='?', const=True,
                    default=True, help="use seed")
parser.add_argument("--use_disp", type=str2bool, nargs='?', const=True,
                    default=False, help="regress towards disparities")
parser.add_argument('--num_samples', default=0, type=int, help='number of samples')

# distributed training
parser.add_argument('--world_size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
parser.add_argument('--local_rank', dest="local_rank", default=0, type=int)


def main():
    global args
    args = parser.parse_args()
    if args.num_samples == 0:
        # Use all lidar points
        args.num_samples = None
    else:
        args.data_path = "" # path to precomputed 500 samples
        assert args.num_samples == 500
        print("changed path to samples500 dataset")
    if args.val_batch_size is None:
        args.val_batch_size = args.batch_size
    if args.seed:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    # init_distributed_mode(args)

    if not args.no_cuda and not torch.cuda.is_available():
        raise Exception("No gpu available for usage")
    torch.backends.cudnn.benchmark = args.cudnn
    # Init model
    args.channels_in = 3 if args.input_type == 'rgb' else 4
    model = Models.define_model(args.mod, args)
    # define_init_weights(model, args.weight_init)

    # Load on gpu before passing params to optimizer
    if not args.no_cuda:
        if not args.multi:
            model = model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
            # model.cuda()
            # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            # model = model.module

    save_id = '{}_{}_{}_{}_{}_batch{}_pretrain{}_wrgb{}_drop{}_patience{}_num_samples{}_multi{}_submod{}'.\
              format(args.mod, args.optimizer, args.loss_criterion,
                     args.learning_rate,
                     args.input_type, 
                     args.batch_size,
                     args.pretrained, args.wrgb, args.drop, 
                     args.lr_decay_iters, args.num_samples, args.multi, args.submod)


    # INIT optimizer/scheduler/loss criterion
    optimizer = define_optim(args.optimizer, model.parameters(), args.learning_rate, args.weight_decay)
    scheduler = define_scheduler(optimizer, args)

    # Optional to use different losses
    criterion_local = define_loss(args.loss_criterion)
    criterion_rgb = define_loss(args.loss_criterion)

    # INIT dataset
    dataset = Datasets.define_dataset(args.dataset, args.data_path, args.input_type)
    dataset.prepare_dataset()
    train_loader, _, valid_loader, valid_selection_loader = get_loader(args, dataset)

    # Resume training
    best_epoch = 0
    lowest_loss = np.inf
    args.save_path = os.path.join(args.save_path, save_id)
    mkdir_if_missing(args.save_path)
    log_file_name = 'log_train_start_0.txt'
    args.resume = first_run(args.save_path)
    if args.resume and not args.test_mode and not args.evaluate:
        path = os.path.join(args.save_path, 'checkpoint_model_epoch_{}.pth.tar'.format(int(args.resume)))
        if os.path.isfile(path):
            log_file_name = 'log_train_start_{}.txt'.format(args.resume)
            # stdout
            sys.stdout = Logger(os.path.join(args.save_path, log_file_name))
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(path)
            args.start_epoch = checkpoint['epoch']
            lowest_loss = checkpoint['loss']
            best_epoch = checkpoint['best epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            log_file_name = 'log_train_start_0.txt'
            # stdout
            sys.stdout = Logger(os.path.join(args.save_path, log_file_name))
            print("=> no checkpoint found at '{}'".format(path))

    # Only evaluate
    elif args.evaluate:
        print("Evaluate only")
        best_file_lst = glob.glob(os.path.join(args.save_path, 'model_best*'))
        if len(best_file_lst) != 0:
            best_file_name = best_file_lst[0]
            print(best_file_name)
            if os.path.isfile(best_file_name):
                sys.stdout = Logger(os.path.join(args.save_path, 'Evaluate.txt'))
                print("=> loading checkpoint '{}'".format(best_file_name))
                checkpoint = torch.load(best_file_name)
                model.load_state_dict(checkpoint['state_dict'])
            else:
                print("=> no checkpoint found at '{}'".format(best_file_name))
        else:
            print("=> no checkpoint found at due to empy list in folder {}".format(args.save_path))
        validate(valid_selection_loader, model, criterion_global, criterion_local)
        return

    # Start training from clean slate
    else:
        # Redirect stdout
        sys.stdout = Logger(os.path.join(args.save_path, log_file_name))

    # INIT MODEL
    print(40*"="+"\nArgs:{}\n".format(args)+40*"=")
    print("Init model: '{}'".format(args.mod))
    print("Number of parameters in model {} is {:.3f}M".format(args.mod.upper(), sum(tensor.numel() for tensor in model.parameters())/1e6))

    # Load pretrained state for cityscapes in GLOBAL net
    # if args.pretrained and not args.resume:
        # if not args.multi:
            # target_state = model.depthnet.state_dict()
        # else:
            # target_state = model.module.depthnet.state_dict()
        # check = torch.load('erfnet_pretrained.pth')
        # for name, val in check.items():
            # # Exclude multi GPU prefix
            # mono_name = name[7:] 
            # if mono_name not in target_state:
                 # continue
            # try:
                # target_state[mono_name].copy_(val)
            # except RuntimeError:
                # continue
        # print('Successfully loaded pretrained model')


    # Create summary writer
    log_path = os.path.join(args.save_path, "logs")
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logger = SummaryWriter(log_path)

    with open(os.path.join(args.save_path, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Start training
    for epoch in range(args.start_epoch, args.nepochs):
        print("\n => Start EPOCH {}".format(epoch + 1))
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print(args.save_path)
        # Adjust learning rate
        if args.lr_policy is not None and args.lr_policy != 'plateau':
            scheduler.step()
            lr = optimizer.param_groups[0]['lr']
            print('lr is set to {}'.format(lr))

        # Define container objects
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        score_train = AverageMeter()
        score_train_1 = AverageMeter()
        metric_train = Metrics(max_depth=args.max_depth, disp=args.use_disp)

        # Train model for args.nepochs
        model.train()

        # compute timing
        end = time.time()

        # Load dataset
        for i, (input, gt) in tqdm(enumerate(train_loader)):

            # Time dataloader
            data_time.update(time.time() - end)

            # Put inputs on gpu if possible
            if not args.no_cuda:
                input, gt = input.cuda(), gt.cuda()
            prediction, hidden = model(input, hidden=(None, None))

            if 'mod' in args.mod or 'stacked' in args.mod:
                loss = criterion_local(prediction[0], gt)
                loss_rgb = criterion_rgb(prediction[1], gt)
                loss += args.wrgb*loss_rgb
                prediction = prediction[0]
            else:
                loss = criterion_local(prediction, gt)

            losses.update(loss.item(), input.size(0))
            metric_train.calculate(prediction[:, 0:1].detach(), gt.detach())
            score_train.update(metric_train.get_metric(args.metric), metric_train.num)
            score_train_1.update(metric_train.get_metric(args.metric_1), metric_train.num)

            # Clip gradients (usefull for instabilities or mistakes in ground truth)
            if args.clip_grad_norm != 0:
                nn.utils.clip_grad_norm(model.parameters(), args.clip_grad_norm)

            # Setup backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Time trainig iteration
            batch_time.update(time.time() - end)
            end = time.time()

            # Print info
            if (i + 1) % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Metric {score.val:.4f} ({score.avg:.4f})'.format(
                       epoch+1, i+1, len(train_loader), batch_time=batch_time,
                       loss=losses,
                       score=score_train))

            batch_num = len(train_loader)

            if ((i + 1) % 10 == 0) and (logger is not None):

                current_step = epoch * batch_num + i

                # Add scalar summaries
                logger.add_scalar('Train_loss/Loss', loss.item(), current_step)

                # utils.record_scalar_summary(result, average_meter, current_step, logger, "Train")
                # # Add system info
                # logger.add_scalar('System/gpu_time', average_meter.average().gpu_time, current_step)
                # logger.add_scalar('System/data_time', average_meter.average().data_time, current_step)

                if ((i + 1) % 200 == 0):

                    # Add some image summary
                    if args.modality == "rgb":
                        input_images = input.cpu()
                    else:
                        input_images = input[:, :3, :, :].cpu()
                        input_depth = torch.unsqueeze(input[:, 3, :, :], dim=1).cpu()

                    rgb_grid = make_grid(input_images[0:6, :, :, :], nrow=3, normalize=True),
                    target_grid = make_grid(gt.cpu()[0:6, :, :, :], nrow=3, normalize=True)
                    pred_grid = make_grid(prediction.cpu()[0:6, :, :, :], nrow=3, normalize=True)

                    logger.add_image('Train/RGB', rgb_grid[0].data.numpy(), current_step)
                    logger.add_image('Train/Depth_gt', target_grid.data.numpy(), current_step)
                    logger.add_image('Train/Depth_pred', pred_grid.data.numpy(), current_step)

                    if args.modality == "rgbd":
                        depth_grid = make_grid(input_depth[0:6, :, :, :], nrow=3, normalize=True)
                        logger.add_image('Train/Depth_input', depth_grid.data.numpy(), current_step)


        print("===> Average RMSE score on training set is {:.4f}".format(score_train.avg))
        print("===> Average MAE score on training set is {:.4f}".format(score_train_1.avg))
        # Evaulate model on validation set
        print("=> Start validation set")
        score_valid, score_valid_1, losses_valid = validate(valid_loader, model, criterion_rgb, criterion_local, epoch)
        print("===> Average RMSE score on validation set is {:.4f}".format(score_valid))
        print("===> Average MAE score on validation set is {:.4f}".format(score_valid_1))
        # Evaluate model on selected validation set
        if args.subset is None:
            print("=> Start selection validation set")
            score_selection, score_selection_1, losses_selection = validate(valid_selection_loader, model, criterion_rgb, criterion_local, epoch)
            total_score = score_selection
            print("===> Average RMSE score on selection set is {:.4f}".format(score_selection))
            print("===> Average MAE score on selection set is {:.4f}".format(score_selection_1))
        else:
            total_score = score_valid

        print("===> Last best score was RMSE of {:.4f} in epoch {}".format(lowest_loss,
                                                                           best_epoch))
        # Adjust lr if loss plateaued
        if args.lr_policy == 'plateau':
            scheduler.step(total_score)
            lr = optimizer.param_groups[0]['lr']
            print('LR plateaued, hence is set to {}'.format(lr))

        # File to keep latest epoch
        with open(os.path.join(args.save_path, 'first_run.txt'), 'w') as f:
            f.write(str(epoch))

        # Save model
        to_save = False
        if total_score < lowest_loss:

            to_save = True
            best_epoch = epoch+1
            lowest_loss = total_score
        save_checkpoint({
            'epoch': epoch + 1,
            'best epoch': best_epoch,
            'arch': args.mod,
            'state_dict': model.state_dict(),
            'loss': lowest_loss,
            'optimizer': optimizer.state_dict()}, to_save, epoch)
    if not args.no_tb:
        writer.close()


def validate(loader, model, criterion_rgb, criterion_local, epoch=0):
    # batch_time = AverageMeter()
    losses = AverageMeter()
    metric = Metrics(max_depth=args.max_depth, disp=args.use_disp)
    score = AverageMeter()
    score_1 = AverageMeter()
    loss_rgb = torch.zeros(1)
    # Evaluate model
    model.eval()
    # Only forward pass, hence no grads needed
    with torch.no_grad():
        # end = time.time()
        for i, (input, gt) in tqdm(enumerate(loader)):
            if not args.no_cuda:
                input, gt = input.cuda(non_blocking=True), gt.cuda(non_blocking=True)
            prediction, hidden = model(input, hidden=(None, None))

            if 'mod' in args.mod or 'stacked' in args.mod:
                loss = criterion_local(prediction[0], gt)
                loss_rgb = criterion_rgb(prediction[1], gt)
                loss += args.wrgb*loss_rgb
                prediction = prediction[0]
            else:
                loss = criterion_local(prediction, gt)

            losses.update(loss.item(), input.size(0))

            metric.calculate(prediction[:, 0:1], gt)
            score.update(metric.get_metric(args.metric), metric.num)
            score_1.update(metric.get_metric(args.metric_1), metric.num)

            # save 8 images for visualization
            skip = 50
            if args.modality == 'd':
                img_merge = None
            else:
                if args.modality == 'rgb':
                    rgb = input
                elif args.modality == 'rgbd':
                    rgb = input[:, :3, :, :]
                    depth = input[:, 3, :, :]

                if i == 0:
                    if args.modality == 'rgbd':
                        img_merge = merge_into_row_with_gt(rgb, depth, gt, prediction)
                    else:
                        img_merge = merge_into_row(rgb, gt, prediction)
                elif (i < 8 * skip) and (i % skip == 0):
                    if args.modality == 'rgbd':
                        row = merge_into_row_with_gt(rgb, depth, gt, prediction)
                    else:
                        row = merge_into_row(rgb, gt, prediction)
                    img_merge = add_row(img_merge, row)
                elif i == 8 * skip:
                    filename = args.save_path + '/comparison_' + str(epoch) + '.png'
                    save_image(img_merge, filename)


            if (i + 1) % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Metric {score.val:.4f} ({score.avg:.4f})'.format(
                       i+1, len(loader), loss=losses,
                       score=score))

        if args.evaluate:
            print("===> Average RMSE score on validation set is {:.4f}".format(score.avg))
            print("===> Average MAE score on validation set is {:.4f}".format(score_1.avg))
    return score.avg, score_1.avg, losses.avg


def save_checkpoint(state, to_copy, epoch):
    filepath = os.path.join(args.save_path, 'checkpoint_model_epoch_{}.pth.tar'.format(epoch))
    torch.save(state, filepath)
    if to_copy:
        if epoch > 0:
            lst = glob.glob(os.path.join(args.save_path, 'model_best*'))
            if len(lst) != 0:
                os.remove(lst[0])
        shutil.copyfile(filepath, os.path.join(args.save_path, 'model_best_epoch_{}.pth.tar'.format(epoch)))
        print("Best model copied")
    if epoch > 0:
        prev_checkpoint_filename = os.path.join(args.save_path, 'checkpoint_model_epoch_{}.pth.tar'.format(epoch-1))
        if os.path.exists(prev_checkpoint_filename):
            os.remove(prev_checkpoint_filename)

if __name__ == '__main__':
    main()
