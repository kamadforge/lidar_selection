import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
import torch
from PIL import Image
import random
import torchvision.transforms.functional as F
from Utils.utils import depth_read


def get_loader(args, dataset):
    """
    Define the different dataloaders for training and validation
    """
    crop_size = (args.crop_h, args.crop_w)
    perform_transformation = not args.no_aug

    train_dataset = Dataset_loader(
            args.data_path, dataset.train_paths, args.input_type, resize=None,
            rotate=args.rotate, crop=crop_size, flip=args.flip, rescale=args.rescale,
            max_depth=args.max_depth, sparse_val=args.sparse_val, normal=args.normal, 
            disp=args.use_disp, train=perform_transformation, num_samples=args.num_samples)
    val_dataset = Dataset_loader(
            args.data_path, dataset.val_paths, args.input_type, resize=None,
            rotate=args.rotate, crop=crop_size, flip=args.flip, rescale=args.rescale,
            max_depth=args.max_depth, sparse_val=args.sparse_val, normal=args.normal, 
            disp=args.use_disp, train=False, num_samples=args.num_samples)
    val_select_dataset = Dataset_loader(
            args.data_path, dataset.selected_paths, args.input_type,
            resize=None, rotate=args.rotate, crop=crop_size,
            flip=args.flip, rescale=args.rescale, max_depth=args.max_depth,
            sparse_val=args.sparse_val, normal=args.normal, 
            disp=args.use_disp, train=False, num_samples=args.num_samples)

    train_sampler = None
    val_sampler = None
    val_select_sampler = None
    if args.subset is not None:
        # use subset for training
        random.seed(1)
        train_idx = [i for i in random.sample(range(len(train_dataset)-1), args.subset)]
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
        random.seed(1)
        val_idx = [i for i in random.sample(range(len(val_dataset)-1), round(args.subset*0.5))]
        val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_idx)

    elif args.multi and args.world_size>1:
        # use distributed training 
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        val_select_sampler = torch.utils.data.distributed.DistributedSampler(val_select_dataset)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        shuffle=train_sampler is None, num_workers=args.nworkers,
        pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_dataset, batch_size=int(args.val_batch_size), sampler=val_sampler,
        shuffle=False, num_workers=args.nworkers_val,
        pin_memory=True, drop_last=True)
    val_selection_loader = DataLoader(
        val_select_dataset, batch_size=1, sampler=val_select_sampler, 
        shuffle=False, num_workers=args.nworkers_val, 
        pin_memory=True, drop_last=False)
    return train_loader, train_sampler, val_loader, val_selection_loader


class Dataset_loader(Dataset):
    """Dataset loader"""

    def __init__(self, data_path, dataset_type, input_type, resize,
                 rotate, crop, flip, rescale, max_depth, sparse_val=0.0, 
                 normal=False, disp=False, train=False, num_samples=None):

        # Constants
        self.use_rgb = input_type == 'rgb'
        self.datapath = data_path
        self.dataset_type = dataset_type
        self.train = train
        self.resize = resize
        self.flip = flip
        self.crop = crop
        self.rotate = rotate
        self.rescale = rescale
        self.max_depth = max_depth
        self.sparse_val = sparse_val
        self.normal = normal

        # Transformations
        self.totensor = transforms.ToTensor()
        self.center_crop = transforms.CenterCrop(size=crop)

        # Names
        self.img_name = 'img'
        self.lidar_name = 'lidar_in' 
        self.gt_name = 'gt' 

        # Define random sampler
        self.num_samples = num_samples
        if num_samples is not None:
            self.random_sampler = Random_Sampler(num_samples)

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])


    def __len__(self):
        """
        Conventional len method
        """
        return len(self.dataset_type['lidar_in'])


    def define_transforms(self, input, gt, img=None):
        # Define random variabels
        hflip_input = np.random.uniform(0.0, 1.0) > 0.5 and self.flip == 'hflip'

        if self.train:
            i, j, h, w = transforms.RandomCrop.get_params(input, output_size=self.crop)
            input = F.crop(input, i, j, h, w)
            gt = F.crop(gt, i, j, h, w)
            if hflip_input:
                input, gt = F.hflip(input), F.hflip(gt)

            if self.use_rgb:
                img = F.crop(img, i, j, h, w)
                if hflip_input:
                    img = F.hflip(img)
            input, gt = depth_read(input, self.sparse_val), depth_read(gt, self.sparse_val)
            # Downsample on the fly
            # if self.num_samples is not None:
                # input = self.random_sampler.sample(input)
        else:
            input, gt = self.center_crop(input), self.center_crop(gt)
            if self.use_rgb:
                img = self.center_crop(img)
            input, gt = depth_read(input, self.sparse_val), depth_read(gt, self.sparse_val)
            # Downsample on the fly
            # if self.num_samples is not None:
                # input = self.random_sampler.sample(input)

        return input, gt, img

    def __getitem__(self, idx):
        """
        Args: idx (int): Index of images to make batch
        Returns (tuple): Sample of velodyne data and ground truth.
        """
        sparse_depth_name = os.path.join(self.dataset_type[self.lidar_name][idx])
        gt_name = os.path.join(self.dataset_type[self.gt_name][idx])
        with open(sparse_depth_name, 'rb') as f:
            sparse_depth = Image.open(f)
            w, h = sparse_depth.size
            sparse_depth = F.crop(sparse_depth, h-self.crop[0], 0, self.crop[0], w)
        with open(gt_name, 'rb') as f:
            gt = Image.open(f)
            gt = F.crop(gt, h-self.crop[0], 0, self.crop[0], w)
        img = None
        if self.use_rgb:
            img_name = self.dataset_type[self.img_name][idx]
            with open(img_name, 'rb') as f:
                img = (Image.open(f).convert('RGB'))
            img = F.crop(img, h-self.crop[0], 0, self.crop[0], w)

        sparse_depth_np, gt_np, img_pil = self.define_transforms(sparse_depth, gt, img)
        input, gt = self.totensor(sparse_depth_np).float(), self.totensor(gt_np).float()

        if self.use_rgb:
            if not self.normal:
                img_tensor = self.totensor(img_pil).float()
                img_tensor = img_tensor*255
            else:
                img_tensor = self.normalize(self.totensor(img_pil).float())

            input = torch.cat((img_tensor, input), dim=0)
        return input, gt


class Random_Sampler():
    "Class to downsample input lidar points"

    def __init__(self, num_samples):
        self.num_samples = num_samples

    def sample(self, depth):
        mask_keep = depth > 0
        n_keep = np.count_nonzero(mask_keep)

        if n_keep == 0:
            return mask_keep
        else:
            depth_sampled = np.zeros(depth.shape)
            prob = float(self.num_samples) / n_keep
            mask_keep =  np.bitwise_and(mask_keep, np.random.uniform(0, 1, depth.shape) < prob)
            depth_sampled[mask_keep] = depth[mask_keep]
            return depth_sampled
