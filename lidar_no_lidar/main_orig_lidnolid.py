import argparse
import os
import time

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torchvision import transforms

from dataloaders.kitti_loader import load_calib, oheight, owidth, input_options, KittiDepth
#from model import DepthCompletionNet
from metrics import AverageMeter, Result
import criteria
import helper
import networks
#from inverse_warp import Intrinsics, homography_from

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='Sparse-to-Dense')
parser.add_argument('-w',
                    '--workers',
                    default=0, #4
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
                    default=20,
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
                    default=100,
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
                    default='rgbd', #'gd'
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
print(torch.cuda.get_device_name(torch.cuda.current_device()))

# define loss functions
depth_criterion = criteria.MaskedMSELoss() if (
    args.criterion == 'l2') else criteria.MaskedL1Loss()
photometric_criterion = criteria.PhotometricLoss()
smoothness_criterion = criteria.SmoothnessLoss()


#
# #### train
#
# def train(self):
#     """Run the entire training pipeline
#     """
#     self.epoch = 0
#     self.step = 0
#     self.start_time = time.time()
#     for self.epoch in range(self.opt.num_epochs):
#         run_epoch()
#         if (self.epoch + 1) % self.opt.save_frequency == 0:
#             self.save_model()
#
#
# def run_epoch(self):
#     """Run a single epoch of training and validation
#     """
#     self.model_lr_scheduler.step()
#
#     print("Training")
#     self.set_train()
#
#     for batch_idx, inputs in enumerate(self.train_loader):
#
#         before_op_time = time.time()
#         outputs, losses = self.process_batch(inputs)
#         self.model_optimizer.zero_grad()
#         losses["loss"].backward()
#         self.model_optimizer.step()
#
#         duration = time.time() - before_op_time
#
#         # log less frequently after the first 2000 steps to save time & disk space
#         early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
#         late_phase = self.step % 2000 == 0
#
#         if early_phase or late_phase:
#             self.log_time(batch_idx, duration, losses["loss"].cpu().data)
#
#             if "depth_gt" in inputs:
#                 self.compute_depth_losses(inputs, outputs, losses)
#
#             self.log("train", inputs, outputs, losses)
#             self.val()
#
#         self.step += 1
#
#
# def process_batch(self, inputs):
#     """Pass a minibatch through the network and generate images and losses
#     """
#     for key, ipt in inputs.items():
#         inputs[key] = ipt.to(self.device)
#
#     # Otherwise, we only feed the image with frame_id 0 through the depth encoder
#     features = self.models["encoder"](inputs["color_aug", 0, 0])
#     outputs = self.models["depth"](features)
#
#     if self.opt.predictive_mask:
#         outputs["predictive_mask"] = self.models["predictive_mask"](features)
#
#     if self.use_pose_net:
#         outputs.update(self.predict_poses(inputs, features))
#
#     self.generate_images_pred(inputs, outputs)
#     losses = self.compute_losses(inputs, outputs)
#
#     return outputs, losses


def iterate(mode, args, loader, model, optimizer, logger, epoch):
    block_average_meter = AverageMeter()
    average_meter = AverageMeter()
    meters = [block_average_meter, average_meter]

    # switch to appropriate mode
    assert mode in ["train", "val", "eval", "test_prediction", "test_completion"], \
        "unsupported mode: {}".format(mode)

    encoder = model[0]; decoder = model[1]
    if mode == 'train':
        encoder.train(); decoder.train()
        lr = helper.adjust_learning_rate(args.lr, optimizer, epoch)
    else:
        encoder.train(); decoder.train()
        lr = 0

    torch.set_printoptions(profile="full")
    for i, batch_data in enumerate(loader):

        start = time.time()
        batch_data = {
            key: val.to(device)
            for key, val in batch_data.items() if val is not None
        }
        gt = batch_data['gt'] if mode != 'test_prediction' and mode != 'test_completion' else None
        data_time = time.time() - start


        start = time.time()

        # torchvision.transforms.Resize
        # img = transform.resize(batch_data['rgb'], (192, 640))

        # transform = transforms.Compose([
        #     transforms.Resize((round(192), round(640))),
        #     # interpolation `BILINEAR` is applied by default
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])
        # image = transform(batch_data['rgb'])



        # Otherwise, we only feed the image with frame_id 0 through the depth encoder
        features = encoder(batch_data['rgb'])
        outputs = decoder(features)

        # if i % 20 == 0:
        #     print(outputs[('disp', 0)][1,0,:,40])

        # pred = model(batch_data)
        # im = batch_data['d'].detach().cpu().numpy()
        # im_sq = im.squeeze()
        # plt.figure()
        # plt.imshow(im_sq)
        # plt.show()
        # for i in range(im_sq.shape[0]):
        #     print(f"{i} - {np.sum(im_sq[i])}")
        pred = outputs[('disp', 0)]
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
                # if i % 20 == 0:
                #     print("\n\n\n gt \n")
                #     print(gt[1,0,:,40])

            # # Loss 2: the self-supervised photometric loss
            # if args.use_pose:
            #     # create multi-scale pyramids
            #     pred_array = helper.multiscale(pred)
            #     rgb_curr_array = helper.multiscale(batch_data['rgb'])
            #     rgb_near_array = helper.multiscale(batch_data['rgb_near'])
            #     if mask is not None:
            #         mask_array = helper.multiscale(mask)
            #     num_scales = len(pred_array)
            #
            #     # compute photometric loss at multiple scales
            #     for scale in range(len(pred_array)):
            #         pred_ = pred_array[scale]
            #         rgb_curr_ = rgb_curr_array[scale]
            #         rgb_near_ = rgb_near_array[scale]
            #         mask_ = None
            #         if mask is not None:
            #             mask_ = mask_array[scale]
            #
            #         # compute the corresponding intrinsic parameters
            #         height_, width_ = pred_.size(2), pred_.size(3)
            #         intrinsics_ = kitti_intrinsics.scale(height_, width_)
            #
            #         # inverse warp from a nearby frame to the current frame
            #         warped_ = homography_from(rgb_near_, pred_,
            #                                   batch_data['r_mat'],
            #                                   batch_data['t_vec'], intrinsics_)
            #         photometric_loss += photometric_criterion(
            #             rgb_curr_, warped_, mask_) * (2**(scale - num_scales))

            # # Loss 3: the depth smoothness loss
            # smooth_loss = smoothness_criterion(pred) if args.w2 > 0 else 0

            # backprop
            #loss = depth_loss + args.w1 * photometric_loss + args.w2 * smooth_loss
            loss = depth_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        gpu_time = time.time() - start


        if i % 50 == 0:
            print(i)

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

        if i % 200 ==0:

            print(gpu_time)


            print("saving")
            avg = logger.conditional_save_info(mode, average_meter, epoch)
            is_best = logger.rank_conditional_save_best(mode, avg, epoch)
            if is_best and not (mode == "train"):
                logger.save_img_comparison_as_best(mode, epoch)
            logger.conditional_summarize(mode, avg, is_best)

            # helper.save_checkpoint({  # save checkpoint
            #     'epoch': epoch,
            #     'model': model.module.state_dict(),
            #     'best_result': logger.best_result,
            #     'optimizer': optimizer.state_dict(),
            #     'args': args,
            # }, is_best, epoch, logger.output_directory)

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

    ################# model

    print("=> creating model and optimizer ... ", end='')
    parameters_to_train = []
    encoder = networks.ResnetEncoder(num_layers=18)
    encoder.to(device)
    parameters_to_train += list(encoder.parameters())
    decoder = networks.DepthDecoder(encoder.num_ch_enc)
    decoder.to(device)
    parameters_to_train += list(decoder.parameters())
    # encoder_named_params = [
    #     p for _, p in encoder.named_parameters() if p.requires_grad
    # ]
    optimizer = torch.optim.Adam(parameters_to_train, lr=args.lr, weight_decay=args.weight_decay)
    encoder = torch.nn.DataParallel(encoder)
    decoder = torch.nn.DataParallel(decoder)
    model = [encoder, decoder]
    print("completed.")
    # if checkpoint is not None:
    #     model.load_state_dict(checkpoint['model'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     print("=> checkpoint state loaded.")


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
        batch_size=12, #1
        shuffle=False,
        num_workers=2,
        pin_memory=True)  # set batch size to be 1 for validation
    print("\t==> val_loader size:{}".format(len(val_loader)))

    ##############################################################

    # create backups and results folder
    logger = helper.logger(args)
    # if checkpoint is not None:
    #     logger.best_result = checkpoint['best_result']
    print("=> logger created.")


    if is_eval:
        print("=> starting model evaluation ...")
        result, is_best = iterate("val", args, val_loader, model, None, logger,checkpoint['epoch'])
        return

    # main loop
    print("=> starting main loop ...")
    for epoch in range(args.start_epoch, args.epochs):
        print("=> starting training epoch {} ..".format(epoch))
        iterate("train", args, train_loader, model, optimizer, logger,
                epoch)  # train for one epoch
        result, is_best = iterate("val", args, val_loader, model, None, logger,
                                  epoch)  # evaluate on validation set
        # helper.save_checkpoint({ # save checkpoint
        #     'epoch': epoch,
        #     'model': model.module.state_dict(),
        #     'best_result': logger.best_result,
        #     'optimizer' : optimizer.state_dict(),
        #     'args' : args,
        # }, is_best, epoch, logger.output_directory)


if __name__ == '__main__':
    main()
