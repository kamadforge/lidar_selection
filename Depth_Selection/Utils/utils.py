#!/usr/bin/python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (35, 30)
from PIL import Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
import os
import torch.optim
from torch.optim import lr_scheduler
import errno
import sys
from torchvision import transforms
import torch.nn.init as init
import torch.distributed as dist
from collections import namedtuple

cmap = plt.cm.jet

def define_optim(optim, params, lr, weight_decay):
    if optim == 'adam':
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optim == 'sgd':
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optim == 'rmsprop':
        optimizer = torch.optim.RMSprop(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise KeyError("The requested optimizer: {} is not implemented".format(optim))
    return optimizer


def define_scheduler(optimizer, args):
    if args.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 - args.niter) / float(args.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer,
                                        step_size=args.lr_decay_iters, gamma=args.gamma)
    elif args.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                   factor=args.gamma,
                                                   threshold=0.0001,
                                                   patience=args.lr_decay_iters)
    elif args.lr_policy == 'none':
        scheduler = None
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler


def define_init_weights(model, init_w='normal', activation='relu'):
    print('Init weights in network with [{}]'.format(init_w))
    if init_w == 'normal':
        model.apply(weights_init_normal)
    elif init_w == 'xavier':
        model.apply(weights_init_xavier)
    elif init_w == 'kaiming':
        model.apply(weights_init_kaiming)
    elif init_w == 'orthogonal':
        model.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [{}] is not implemented'.format(init_w))


def first_run(save_path):
    txt_file = os.path.join(save_path, 'first_run.txt')
    if not os.path.exists(txt_file):
        open(txt_file, 'w').close()
    else:
        saved_epoch = open(txt_file).read()
        if saved_epoch is None:
            print('You forgot to delete [first run file]')
            return ''
        return saved_epoch
    return ''


def depth_read(img, sparse_val):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt
    depth_png = np.array(img, dtype=int)
    depth_png = np.expand_dims(depth_png, axis=2)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert(np.max(depth_png) > 255)
    depth = depth_png.astype(np.float) / 256.
    depth[depth_png == 0] = sparse_val
    return depth


class show_figs():
    def __init__(self, input_type, savefig=False):
        self.input_type = input_type
        self.savefig = savefig

    def save(self, img, name):
        img.save(name)

    def transform(self, input, name='test.png'):
        if isinstance(input, torch.tensor):
            input = torch.clamp(input, min=0, max=255).int().cpu().numpy()
            input = input * 256.
            img = Image.fromarray(input)

        elif isinstance(input, np.array):
            img = Image.fromarray(input)

        else:
            raise NotImplementedError('Input type not recognized type')

        if self.savefig:
            self.save(img, name)
        else:
            return img

def str2bool(argument):
    if argument.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif argument.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Wrong argument in argparse, should be a boolean')


def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def synchronize_between_processes(self):
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.sum], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.sum = t[1]
        self.avg = self.sum / self.count


def write_file(content, location):
    file = open(location, 'w')
    file.write(str(content))
    file.close()


class Logger(object):
    """
    Source https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        self.fpath = fpath
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

def save_image(img_merge, filename):
    img_merge = Image.fromarray(img_merge.astype('uint8'))
    img_merge.save(filename)

def get_pose(img_path):
    """
    Input RGB Image path
    """

    # Per dataformat.txt
    OxtsPacket = namedtuple('OxtsPacket',
                            'lat, lon, alt, ' +
                            'roll, pitch, yaw, ' +
                            'vn, ve, vf, vl, vu, ' +
                            'ax, ay, az, af, al, au, ' +
                            'wx, wy, wz, wf, wl, wu, ' +
                            'pos_accuracy, vel_accuracy, ' +
                            'navstat, numsats, ' +
                            'posmode, velmode, orimode')

    # Bundle into an easy-to-access structure
    OxtsData = namedtuple('OxtsData', 'packet, T_w_imu')

    # Scale for Mercator projection (from first lat value)
    scale = None

    # Origin of the global coordinate system (first GPS position)
    origin = [0,0,0] #None

    oxts = []

    oxts_file = os.path.abspath(os.path.join(img_path ,"../../..",'oxts/data', os.path.basename(img_path)[:-4] + '.txt'))

    with open(os.path.join(oxts_file), 'r') as f:
        for line in f.readlines():

            line = line.split()
            # Last five entries are flags and counts
            line[:-5] = [float(x) for x in line[:-5]]
            line[-5:] = [int(float(x)) for x in line[-5:]]

            packet = OxtsPacket(*line)

            if scale is None:
                scale = np.cos(packet.lat * np.pi / 180.)

            er = 6378137.  # earth radius (approx.) in meters

            # Use a Mercator projection to get the translation vector
            tx = scale * packet.lon * np.pi * er / 180.
            ty = scale * er * \
                np.log(np.tan((90. + packet.lat) * np.pi / 360.))
            tz = packet.alt
            t = np.array([tx, ty, tz])
            t = t.reshape(3, 1)


            Rx = np.array([[1,   0,                    0                  ],
                            [0,  np.cos(packet.roll), -np.sin(packet.roll)],
                            [0,  np.sin(packet.roll),  np.cos(packet.roll)]])
            Ry = np.array([[np.cos(packet.pitch),  0,  np.sin(packet.pitch)],
                             [0,                   1,                       0],
                             [-np.sin(packet.pitch), 0,  np.cos(packet.pitch)]])
            Rz = np.array([[np.cos(packet.yaw), -np.sin(packet.yaw),  0],
                             [np.sin(packet.yaw),  np.cos(packet.yaw),  0],
                             [0,                   0,                   1]])
            R = Rz.dot(Ry.dot(Rx))
            R = R.reshape(3, 3)

            T_w_imu = np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))

            oxts.append(OxtsData(packet, T_w_imu))

    pose = oxts[0][1]
    return pose

def get_intrinsics(img_path):
    """
    Input RGB Image path
    """

    P_line_in_file = [7, 15, 23, 31]

    if "image_02" in img_path:
        cam = 2
    elif "image_03" in img_path:
        cam = 3

    filepath_cam_to_cam =os.path.abspath(os.path.join(img_path ,"../../../..",'calib_cam_to_cam.txt'))

    with open(filepath_cam_to_cam, 'r') as f:
        all_matrices = [row.strip().split() for row in f]

    for row in all_matrices:
        del row[0]

    all_matrices = [x for x in all_matrices[2:] if x != []]
    all_matrices = np.array([np.array(xi, dtype=float) for xi in all_matrices])

    intrinsics = all_matrices[P_line_in_file[cam]]
    intrinsics = intrinsics.reshape((3, 4)) #[:, :3]
    intrinsics = np.vstack((intrinsics, [0, 0, 0, 1]))
    
    return intrinsics


def weights_init_normal(m):
    classname = m.__class__.__name__
#    print(classname)
    if classname.find('Conv') != -1 or classname.find('ConvTranspose') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1 or classname.find('ConvTranspose') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1 or classname.find('ConvTranspose') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
#    print(classname)
    if classname.find('Conv') != -1 or classname.find('ConvTranspose') != -1:
        init.orthogonal(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def save_fig(inp, name='saved.png'):
    if isinstance(inp, torch.Tensor):
        # inp = inp.permute([2, 0, 1])
        inp = transforms.ToPILImage()(inp.int())
        inp.save(name)
        return
    pil = Image.fromarray(inp)
    pil.save(name)

def add_row(img_merge, row):
    return np.vstack([img_merge, row])

def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:,:,:3] # H, W, C

def merge_into_row(input, depth_target, depth_pred):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1, 2, 0))  # H, W, C
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

    d_min = min(np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_target_cpu), np.max(depth_pred_cpu))
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)
    img_merge = np.hstack([rgb, depth_target_col, depth_pred_col])
    return img_merge

def merge_into_row_with_gt(input, depth_input, depth_target, depth_pred):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1, 2, 0))  # H, W, C
    depth_input_cpu = np.squeeze(depth_input.cpu().numpy())
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

    d_min = min(np.min(depth_input_cpu), np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_input_cpu), np.max(depth_target_cpu), np.max(depth_pred_cpu))
    depth_input_col = colored_depthmap(depth_input_cpu, d_min, d_max)
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)

    img_merge = np.hstack([rgb, depth_input_col, depth_target_col, depth_pred_col])

    return img_merge

########################
# Distributed training #
########################

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    # Does not seem to work?
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)
