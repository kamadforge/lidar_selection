# MODELS
# DepthCompletionNet - original
# DepthCompletionNetQSquare - global squares
# DepthCompletionNetQSquareNet - local squares
# DepthCompletionNetQ - global lines (DepthCompletionNetQFit is a copy)



import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
import matplotlib.pyplot as plt
from torch.nn.parameter import Parameter
import numpy as np

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.set_printoptions(profile="full")



def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 1e-3)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        m.weight.data.normal_(0, 1e-3)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

def conv_bn_relu(in_channels, out_channels, kernel_size, \
        stride=1, padding=0, bn=True, relu=True):
    bias = not bn
    layers = []
    layers.append(
        nn.Conv2d(in_channels,
                  out_channels,
                  kernel_size,
                  stride,
                  padding,
                  bias=bias))
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    layers = nn.Sequential(*layers)

    # initialize the weights
    for m in layers.modules():
        init_weights(m)

    return layers

def lin_bn_relu(in_channels, out_channels, bn=False, relu=True):
    bias = not bn
    layers = []
    layers.append(
        nn.Linear(in_channels,
                  out_channels))
    if bn:
        layers.append(nn.BatchNorm(out_channels))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    layers = nn.Sequential(*layers)

    # initialize the weights
    for m in layers.modules():
        init_weights(m)

    return layers

def convt_bn_relu(in_channels, out_channels, kernel_size, \
        stride=1, padding=0, output_padding=0, bn=True, relu=True):
    bias = not bn
    layers = []
    layers.append(
        nn.ConvTranspose2d(in_channels,
                           out_channels,
                           kernel_size,
                           stride,
                           padding,
                           output_padding,
                           bias=bias))
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    layers = nn.Sequential(*layers)

    # initialize the weights
    for m in layers.modules():
        init_weights(m)

    return layers

class DepthCompletionNet(nn.Module):
    def __init__(self, args):
        assert (
            args.layers in [18, 34, 50, 101, 152]
        ), 'Only layers 18, 34, 50, 101, and 152 are defined, but got {}'.format(
            layers)
        super(DepthCompletionNet, self).__init__()
        self.modality = args.input

        if 'd' in self.modality:
            channels = 64 // len(self.modality)
            self.conv1_d = conv_bn_relu(1,
                                        channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        if 'rgb' in self.modality:
            channels = 64 * 3 // len(self.modality)
            self.conv1_img = conv_bn_relu(3,
                                          channels,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1)
        elif 'g' in self.modality:
            channels = 64 // len(self.modality)
            self.conv1_img = conv_bn_relu(1,
                                          channels,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1)

        pretrained_model = resnet.__dict__['resnet{}'.format(
            args.layers)](pretrained=args.pretrained)
        if not args.pretrained:
            pretrained_model.apply(init_weights)
        #self.maxpool = pretrained_model._modules['maxpool']
        self.conv2 = pretrained_model._modules['layer1']
        self.conv3 = pretrained_model._modules['layer2']
        self.conv4 = pretrained_model._modules['layer3']
        self.conv5 = pretrained_model._modules['layer4']
        del pretrained_model  # clear memory

        # define number of intermediate channels
        if args.layers <= 34:
            num_channels = 512
        elif args.layers >= 50:
            num_channels = 2048
        self.conv6 = conv_bn_relu(num_channels,
                                  512,
                                  kernel_size=3,
                                  stride=2,
                                  padding=1)

        # decoding layers
        kernel_size = 3
        stride = 2
        self.convt5 = convt_bn_relu(in_channels=512,
                                    out_channels=256,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=1,
                                    output_padding=1)
        self.convt4 = convt_bn_relu(in_channels=768,
                                    out_channels=128,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=1,
                                    output_padding=1)
        self.convt3 = convt_bn_relu(in_channels=(256 + 128),
                                    out_channels=64,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=1,
                                    output_padding=1)
        self.convt2 = convt_bn_relu(in_channels=(128 + 64),
                                    out_channels=64,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=1,
                                    output_padding=1)
        self.convt1 = convt_bn_relu(in_channels=128,
                                    out_channels=64,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=1)
        self.convtf = conv_bn_relu(in_channels=128,
                                   out_channels=1,
                                   kernel_size=1,
                                   stride=1,
                                   bn=False,
                                   relu=False)

    def forward(self, x):
        # first layer
        if 'd' in self.modality:
            print(f"depth input to the network: {len(torch.where(x['d']>0)[0])}")
            conv1_d = self.conv1_d(x['d'])
        if 'rgb' in self.modality:
            conv1_img = self.conv1_img(x['rgb'])
        elif 'g' in self.modality:
            conv1_img = self.conv1_img(x['g'])

        if self.modality == 'rgbd' or self.modality == 'gd':
            conv1 = torch.cat((conv1_d, conv1_img), 1)
        else:
            conv1 = conv1_d if (self.modality == 'd') else conv1_img

        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)  # batchsize * ? * 176 * 608
        conv4 = self.conv4(conv3)  # batchsize * ? * 88 * 304
        conv5 = self.conv5(conv4)  # batchsize * ? * 44 * 152
        conv6 = self.conv6(conv5)  # batchsize * ? * 22 * 76

        # decoder
        convt5 = self.convt5(conv6)
        y = torch.cat((convt5, conv5), 1)

        convt4 = self.convt4(y)
        y = torch.cat((convt4, conv4), 1)

        convt3 = self.convt3(y)
        y = torch.cat((convt3, conv3), 1)

        convt2 = self.convt2(y)
        y = torch.cat((convt2, conv2), 1)

        convt1 = self.convt1(y)
        y = torch.cat((convt1, conv1), 1)

        y = self.convtf(y)

        if self.training:
            return 100 * y
        else:
            min_distance = 0.9
            return F.relu(
                100 * y - min_distance
            ) + min_distance  # the minimum range of Velodyne is around 3 feet ~= 0.9m

class DepthCompletionNetQ(nn.Module):
    def __init__(self, args):
        assert (
            args.layers in [18, 34, 50, 101, 152]
        ), 'Only layers 18, 34, 50, 101, and 152 are defined, but got {}'.format(
            layers)
        super(DepthCompletionNetQ, self).__init__()
        self.modality = args.input

        num = 352
        num = 65
        self.parameter = Parameter(-1e-10 * torch.ones((num)), requires_grad=True)
        self.parameter_mask = torch.Tensor(np.load("features/kitti_pixels_to_lines_masks.npy", allow_pickle=True)).to(device)
        self.phi = None

        if 'd' in self.modality:
            channels = 64 // len(self.modality)
            self.conv1_d = conv_bn_relu(1,
                                        channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        if 'rgb' in self.modality:
            channels = 64 * 3 // len(self.modality)
            self.conv1_img = conv_bn_relu(3,
                                          channels,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1)
        elif 'g' in self.modality:
            channels = 64 // len(self.modality)
            self.conv1_img = conv_bn_relu(1,
                                          channels,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1)

        pretrained_model = resnet.__dict__['resnet{}'.format(
            args.layers)](pretrained=args.pretrained)
        if not args.pretrained:
            pretrained_model.apply(init_weights)
        #self.maxpool = pretrained_model._modules['maxpool']
        self.conv2 = pretrained_model._modules['layer1']
        self.conv3 = pretrained_model._modules['layer2']
        self.conv4 = pretrained_model._modules['layer3']
        self.conv5 = pretrained_model._modules['layer4']
        del pretrained_model  # clear memory

        # define number of intermediate channels
        if args.layers <= 34:
            num_channels = 512
        elif args.layers >= 50:
            num_channels = 2048
        self.conv6 = conv_bn_relu(num_channels,
                                  512,
                                  kernel_size=3,
                                  stride=2,
                                  padding=1)

        # decoding layers
        kernel_size = 3
        stride = 2
        self.convt5 = convt_bn_relu(in_channels=512,
                                    out_channels=256,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=1,
                                    output_padding=1)
        self.convt4 = convt_bn_relu(in_channels=768,
                                    out_channels=128,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=1,
                                    output_padding=1)
        self.convt3 = convt_bn_relu(in_channels=(256 + 128),
                                    out_channels=64,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=1,
                                    output_padding=1)
        self.convt2 = convt_bn_relu(in_channels=(128 + 64),
                                    out_channels=64,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=1,
                                    output_padding=1)
        self.convt1 = convt_bn_relu(in_channels=128,
                                    out_channels=64,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=1)
        self.convtf = conv_bn_relu(in_channels=128,
                                   out_channels=1,
                                   kernel_size=1,
                                   stride=1,
                                   bn=False,
                                   relu=False)

    def forward(self, x):

        pre_phi = self.parameter
        # print(self.parameter)
        self.phi = F.softplus(self.parameter)

        # if any(torch.isnan(phi)):
        if any(torch.flatten(torch.isnan(self.phi))):
            print("some Phis are NaN")
        # it looks like too large values are making softplus-transformed values very large and returns NaN.
        # this occurs when optimizing with a large step size (or/and with a high momentum value)

        S = self.phi / torch.sum(self.phi)

        ###

        # pre_phi = self.parameter
        # # print(self.parameter)
        # phi = F.softplus(self.parameter)
        #
        # if any(torch.isnan(phi)):
        #     print("some Phis are NaN")
        # # it looks like too large values are making softplus-transformed values very large and returns NaN.
        # # this occurs when optimizing with a large step size (or/and with a high momentum value)
        #
        # S = phi / torch.sum(phi)

        #########

        # Slen=len(S)
        # S_expand = S.repeat(x['d'].shape[-1]).reshape(Slen, x['d'].shape[-1])
        # output = x['d'] * S_expand

        # switch mask
        S_mask_ext = torch.einsum("i, ijk->ijk", [S, self.parameter_mask])
        print(S_mask_ext[24][308][733])
        S_mask = torch.max(S_mask_ext, 0)[0]

        # for i in range(len(self.parameter)):
        #     print(f"{i}: {len(np.where(S_mask_ind.detach().cpu().numpy() == i)[0])}")
        output = x['d'] * S_mask





        # first layer
        if 'd' in self.modality:
            conv1_d = self.conv1_d(output)
        if 'rgb' in self.modality:
            conv1_img = self.conv1_img(x['rgb'])
        elif 'g' in self.modality:
            conv1_img = self.conv1_img(x['g'])

        if self.modality == 'rgbd' or self.modality == 'gd':
            conv1 = torch.cat((conv1_d, conv1_img), 1)
        else:
            conv1 = conv1_d if (self.modality == 'd') else conv1_img

        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)  # batchsize * ? * 176 * 608
        conv4 = self.conv4(conv3)  # batchsize * ? * 88 * 304
        conv5 = self.conv5(conv4)  # batchsize * ? * 44 * 152
        conv6 = self.conv6(conv5)  # batchsize * ? * 22 * 76

        # decoder
        convt5 = self.convt5(conv6)
        y = torch.cat((convt5, conv5), 1)

        convt4 = self.convt4(y)
        y = torch.cat((convt4, conv4), 1)

        convt3 = self.convt3(y)
        y = torch.cat((convt3, conv3), 1)

        convt2 = self.convt2(y)
        y = torch.cat((convt2, conv2), 1)

        convt1 = self.convt1(y)
        y = torch.cat((convt1, conv1), 1)

        y = self.convtf(y)

        if self.training:
            return 100 * y
        else:
            min_distance = 0.9
            return F.relu(
                100 * y - min_distance
            ) + min_distance  # the minimum range of Velodyne is around 3 feet ~= 0.9m

class DepthCompletionNetQFit(nn.Module):
    def __init__(self, args):
        assert (
            args.layers in [18, 34, 50, 101, 152]
        ), 'Only layers 18, 34, 50, 101, and 152 are defined, but got {}'.format(
            layers)
        super(DepthCompletionNetQFit, self).__init__()
        self.modality = args.input

        ######
        num=352
        num = 65
        self.parameter = Parameter(-1e-10*torch.ones(num),requires_grad=True)
        self.parameter_mask = torch.Tensor(np.load("../kitti_pixels_to_lines.npy", allow_pickle=True)).to(device)
        # self.phi_fc1 = nn.Linear(3, num)
        # self.phi_fc2 = nn.Linear(num, num)
        # # self.phi_fc2b = nn.Linear(200, 200)
        # self.phi_fc3 = nn.Linear(num, 30)  # outputs switch values
        #
        # self.fc1_bn1 = nn.BatchNorm1d(num)
        # # self.fc2_bn2b = nn.BatchNorm1d(200)
        # self.fc2_bn2 = nn.BatchNorm1d(num)
        #####


        if 'd' in self.modality:
            channels = 64 // len(self.modality)
            self.conv1_d = conv_bn_relu(1,
                                        channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        if 'rgb' in self.modality:
            channels = 64 * 3 // len(self.modality)
            self.conv1_img = conv_bn_relu(3,
                                          channels,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1)
        elif 'g' in self.modality:
            channels = 64 // len(self.modality)
            self.conv1_img = conv_bn_relu(1,
                                          channels,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1)

        pretrained_model = resnet.__dict__['resnet{}'.format(
            args.layers)](pretrained=args.pretrained)
        if not args.pretrained:
            pretrained_model.apply(init_weights)
        #self.maxpool = pretrained_model._modules['maxpool']
        self.conv2 = pretrained_model._modules['layer1']
        self.conv3 = pretrained_model._modules['layer2']
        self.conv4 = pretrained_model._modules['layer3']
        self.conv5 = pretrained_model._modules['layer4']
        del pretrained_model  # clear memory

        # define number of intermediate channels
        if args.layers <= 34:
            num_channels = 512
        elif args.layers >= 50:
            num_channels = 2048
        self.conv6 = conv_bn_relu(num_channels,
                                  512,
                                  kernel_size=3,
                                  stride=2,
                                  padding=1)

        # decoding layers
        kernel_size = 3
        stride = 2
        self.convt5 = convt_bn_relu(in_channels=512,
                                    out_channels=256,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=1,
                                    output_padding=1)
        self.convt4 = convt_bn_relu(in_channels=768,
                                    out_channels=128,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=1,
                                    output_padding=1)
        self.convt3 = convt_bn_relu(in_channels=(256 + 128),
                                    out_channels=64,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=1,
                                    output_padding=1)
        self.convt2 = convt_bn_relu(in_channels=(128 + 64),
                                    out_channels=64,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=1,
                                    output_padding=1)
        self.convt1 = convt_bn_relu(in_channels=128,
                                    out_channels=64,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=1)
        self.convtf = conv_bn_relu(in_channels=128,
                                   out_channels=1,
                                   kernel_size=1,
                                   stride=1,
                                   bn=False,
                                   relu=False)

    def forward(self, x):

       #################################

        pre_phi = self.parameter
        #print(self.parameter)
        phi = F.softplus(self.parameter)

        if any(torch.isnan(phi)):
           print("some Phis are NaN")
        # it looks like too large values are making softplus-transformed values very large and returns NaN.
        # this occurs when optimizing with a large step size (or/and with a high momentum value)


        S = phi / torch.sum(phi)
        #print(S)

        # Slen=len(S)
        # S_expand = S.repeat(x['d'].shape[-1]).reshape(Slen, x['d'].shape[-1])
        # output = x['d'] * S_expand

        S_mask_ext = torch.einsum("i, ijk->ijk", [S, self.parameter_mask])
        print(S_mask_ext[24][308][733])
        S_mask=torch.max(S_mask_ext, 0)[0]

        S_mask_ind=torch.max(S_mask_ext, 0)[1]

        #print(S_mask_ind[250])

        # for i in range(len(self.parameter)):
        #     print(f"{i}: {len(np.where(S_mask_ind.detach().cpu().numpy() == i)[0])}")

        #print(imp_lines[1])
        #print(imp_lines[40])

        #print(imp_lines[250])
        #print(imp_lines[300])


        #print(S_mask.shape)
        #print(x['d'].shape)
        output = x['d'] * S_mask


        #
        # output = self.phi_fc1(x['d'])
        # # output = nn.functional.relu(self.phi_fc1(x))
        #
        # output = self.fc1_bn1(output)
        # output = nn.functional.relu(self.phi_fc2(output))
        # output = self.fc2_bn2(output)
        # # output = nn.functional.relu(self.phi_fc2b(output))
        # # output = self.fc2_bn2b(output)
        #
        # pre_phi = self.phi_fc3(output)
        #
        # # phi = F.softplus(phi_parameter.mean(dim=0))
        # phi = F.softplus(pre_phi)  # now the size of phi is mini_batch by input_dim
        #
        # S = phi / torch.sum(phi, dim=1).unsqueeze(dim=1)  # [batch x featnum]
        # output = x['d'] * S

        #################3

        # first layer
        if 'd' in self.modality:
            conv1_d = self.conv1_d(output)
        if 'rgb' in self.modality:
            conv1_img = self.conv1_img(x['rgb'])
        elif 'g' in self.modality:
            conv1_img = self.conv1_img(x['g'])

        if self.modality == 'rgbd' or self.modality == 'gd':
            conv1 = torch.cat((conv1_d, conv1_img), 1)
        else:
            conv1 = conv1_d if (self.modality == 'd') else conv1_img

        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)  # batchsize * ? * 176 * 608
        conv4 = self.conv4(conv3)  # batchsize * ? * 88 * 304
        conv5 = self.conv5(conv4)  # batchsize * ? * 44 * 152
        conv6 = self.conv6(conv5)  # batchsize * ? * 22 * 76

        # decoder
        convt5 = self.convt5(conv6)
        y = torch.cat((convt5, conv5), 1)

        convt4 = self.convt4(y)
        y = torch.cat((convt4, conv4), 1)

        convt3 = self.convt3(y)
        y = torch.cat((convt3, conv3), 1)

        convt2 = self.convt2(y)
        y = torch.cat((convt2, conv2), 1)

        convt1 = self.convt1(y)
        y = torch.cat((convt1, conv1), 1)

        y = self.convtf(y)

        if self.training:
            return 100 * y
        else:
            min_distance = 0.9
            return F.relu(
                100 * y - min_distance
            ) + min_distance  # the minimum range of Velodyne is around 3 feet ~= 0.9m



class DepthCompletionNetQSquare(nn.Module):
    def __init__(self, args):
        assert (
            args.layers in [18, 34, 50, 101, 152]
        ), 'Only layers 18, 34, 50, 101, and 152 are defined, but got {}'.format(
            layers)
        super(DepthCompletionNetQSquare, self).__init__()
        self.modality = args.input

        self.img_height=352
        self.img_width=1216
        self.bin_ver = np.arange(0, self.img_height, 40)
        self.bin_ver = np.append(self.bin_ver, self.img_height)
        self.bin_hor = np.arange(0, self.img_width, 40)
        self.bin_hor = np.append(self.bin_hor, self.img_width)

        num = 352
        num = 65
        #self.parameter = Parameter(-1e-10 * torch.ones(num), requires_grad=True)
        self.parameter = Parameter(-1e-10 * torch.ones(len(self.bin_ver)-1 , len(self.bin_hor)-1))
        self.phi=None
        #self.parameter_mask = torch.Tensor(np.load("../kitti_pixels_to_lines.npy", allow_pickle=True)).to(device)

        if 'd' in self.modality:
            channels = 64 // len(self.modality)
            self.conv1_d = conv_bn_relu(1,
                                        channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        if 'rgb' in self.modality:
            channels = 64 * 3 // len(self.modality)
            self.conv1_img = conv_bn_relu(3,
                                          channels,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1)
        elif 'g' in self.modality:
            channels = 64 // len(self.modality)
            self.conv1_img = conv_bn_relu(1,
                                          channels,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1)

        pretrained_model = resnet.__dict__['resnet{}'.format(
            args.layers)](pretrained=args.pretrained)
        if not args.pretrained:
            pretrained_model.apply(init_weights)
        #self.maxpool = pretrained_model._modules['maxpool']
        self.conv2 = pretrained_model._modules['layer1']
        self.conv3 = pretrained_model._modules['layer2']
        self.conv4 = pretrained_model._modules['layer3']
        self.conv5 = pretrained_model._modules['layer4']
        del pretrained_model  # clear memory

        # define number of intermediate channels
        if args.layers <= 34:
            num_channels = 512
        elif args.layers >= 50:
            num_channels = 2048
        self.conv6 = conv_bn_relu(num_channels,
                                  512,
                                  kernel_size=3,
                                  stride=2,
                                  padding=1)

        # decoding layers
        kernel_size = 3
        stride = 2
        self.convt5 = convt_bn_relu(in_channels=512,
                                    out_channels=256,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=1,
                                    output_padding=1)
        self.convt4 = convt_bn_relu(in_channels=768,
                                    out_channels=128,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=1,
                                    output_padding=1)
        self.convt3 = convt_bn_relu(in_channels=(256 + 128),
                                    out_channels=64,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=1,
                                    output_padding=1)
        self.convt2 = convt_bn_relu(in_channels=(128 + 64),
                                    out_channels=64,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=1,
                                    output_padding=1)
        self.convt1 = convt_bn_relu(in_channels=128,
                                    out_channels=64,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=1)
        self.convtf = conv_bn_relu(in_channels=128,
                                   out_channels=1,
                                   kernel_size=1,
                                   stride=1,
                                   bn=False,
                                   relu=False)

    def forward(self, x):

        pre_phi = self.parameter
        # print(self.parameter)
        self.phi = F.softplus(self.parameter)

        #if any(torch.isnan(phi)):
        if any(torch.flatten(torch.isnan(self.phi))):
            print("some Phis are NaN")
        # it looks like too large values are making softplus-transformed values very large and returns NaN.
        # this occurs when optimizing with a large step size (or/and with a high momentum value)

        S = self.phi / torch.sum(self.phi)

        ### without self, probably doesn't matter
        # pre_phi = self.parameter
        # # print(self.parameter)
        # phi = F.softplus(self.parameter)
        #
        # # if any(torch.isnan(phi)):
        # if any(torch.flatten(torch.isnan(phi))):
        #     print("some Phis are NaN")
        # # it looks like too large values are making softplus-transformed values very large and returns NaN.
        # # this occurs when optimizing with a large step size (or/and with a high momentum value)
        #
        # S = phi / torch.sum(phi)

        ####



        #print("S from model", S)
        #print("parameter from model", self.parameter)

        # Slen=len(S)
        # S_expand = S.repeat(x['d'].shape[-1]).reshape(Slen, x['d'].shape[-1])
        # output = x['d'] * S_expand

        # switch mask
        # S_mask_ext = torch.einsum("i, ijk->ijk", [S, self.parameter_mask])
        # print(S_mask_ext[24][308][733])
        # S_mask = torch.max(S_mask_ext, 0)[0]
        # # for i in range(len(self.parameter)):
        # #     print(f"{i}: {len(np.where(S_mask_ind.detach().cpu().numpy() == i)[0])}")
        # print(x['d'].shape)
        # output = x['d'] * S_mask

        #switch many features
        mask = torch.zeros((self.img_height, self.img_width)).to(device)
        for i in range(len(self.bin_ver)-1):
            for j in range(len(self.bin_hor)-1):
                # print(self.bin_hor[i])
                # print(self.bin_hor[i+1])
                # print(self.bin_hor[j])
                # print(self.bin_hor[j+1])
                mask[self.bin_hor[i]:self.bin_hor[i+1], self.bin_hor[j]:self.bin_hor[j+1]]=S[i,j]#self.parameter[i,j]
        output = x['d'] * mask



        # first layer
        if 'd' in self.modality:
            conv1_d = self.conv1_d(output)
        if 'rgb' in self.modality:
            conv1_img = self.conv1_img(x['rgb'])
        elif 'g' in self.modality:
            conv1_img = self.conv1_img(x['g'])

        if self.modality == 'rgbd' or self.modality == 'gd':
            conv1 = torch.cat((conv1_d, conv1_img), 1)
        else:
            conv1 = conv1_d if (self.modality == 'd') else conv1_img

        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)  # batchsize * ? * 176 * 608
        conv4 = self.conv4(conv3)  # batchsize * ? * 88 * 304
        conv5 = self.conv5(conv4)  # batchsize * ? * 44 * 152
        conv6 = self.conv6(conv5)  # batchsize * ? * 22 * 76

        # decoder
        convt5 = self.convt5(conv6)
        y = torch.cat((convt5, conv5), 1)

        convt4 = self.convt4(y)
        y = torch.cat((convt4, conv4), 1)

        convt3 = self.convt3(y)
        y = torch.cat((convt3, conv3), 1)

        convt2 = self.convt2(y)
        y = torch.cat((convt2, conv2), 1)

        convt1 = self.convt1(y)
        y = torch.cat((convt1, conv1), 1)

        y = self.convtf(y)

        if self.training:
            return 100 * y
        else:
            min_distance = 0.9
            return F.relu(
                100 * y - min_distance
            ) + min_distance  # the minimum range of Velodyne is around 3 feet ~= 0.9m



#######################



class DepthCompletionNetQSquareNet(nn.Module):
    def __init__(self, args):
        assert (
            args.layers in [18, 34, 50, 101, 152]
        ), 'Only layers 18, 34, 50, 101, and 152 are defined, but got {}'.format(
            layers)
        super(DepthCompletionNetQSquareNet, self).__init__()
        self.modality = args.input

        self.img_height=352
        self.img_width=1216
        self.bin_ver = np.arange(0, self.img_height, 40)
        self.bin_ver = np.append(self.bin_ver, self.img_height)
        self.bin_hor = np.arange(0, self.img_width, 40)
        self.bin_hor = np.append(self.bin_hor, self.img_width)

        num = 352
        num = 65
        #self.parameter = Parameter(-1e-10 * torch.ones(num), requires_grad=True)
        self.parameter = Parameter(-1e-10 * torch.ones(len(self.bin_ver)-1 , len(self.bin_hor)-1))
        self.phi = None
        #self.parameter_mask = torch.Tensor(np.load("../kitti_pixels_to_lines.npy", allow_pickle=True)).to(device)

##################### Q-FIT

        if 'd' in self.modality:
            channels = 64 // len(self.modality)
            self.conv1_d_qfit = conv_bn_relu(1,
                                        channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        if 'rgb' in self.modality:
            channels = 64 * 3 // len(self.modality)
            self.conv1_img_qfit = conv_bn_relu(3,
                                          channels,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1)
        elif 'g' in self.modality:
            channels = 64 // len(self.modality)
            self.conv1_img_qfit = conv_bn_relu(1,
                                          channels,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1)

        self.conv2_qfit = conv_bn_relu(64,
                                           channels,
                                           kernel_size=5,
                                           stride=3,
                                           padding=1)

        self.conv3_qfit = conv_bn_relu(channels,
                                       channels,
                                       kernel_size=5,
                                       stride=3,
                                       padding=1)

        self.conv4_qfit = conv_bn_relu(channels,
                                       channels,
                                       kernel_size=5,
                                       stride=2,
                                       padding=1)

        self.conv5_qfit = conv_bn_relu(channels,
                                       1,
                                       kernel_size=5,
                                       stride=2,
                                       padding=1)

        # pretrained_model = resnet.__dict__['resnet{}'.format(
        #     args.layers)](pretrained=args.pretrained)
        # if not args.pretrained:
        #     pretrained_model.apply(init_weights)
        # self.maxpool = pretrained_model._modules['maxpool']
        # self.conv2 = pretrained_model._modules['layer1']
        # self.conv3 = pretrained_model._modules['layer2']


#################################3


        if 'd' in self.modality:
            channels = 64 // len(self.modality)
            self.conv1_d = conv_bn_relu(1,
                                        channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        if 'rgb' in self.modality:
            channels = 64 * 3 // len(self.modality)
            self.conv1_img = conv_bn_relu(3,
                                          channels,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1)
        elif 'g' in self.modality:
            channels = 64 // len(self.modality)
            self.conv1_img = conv_bn_relu(1,
                                          channels,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1)

        pretrained_model = resnet.__dict__['resnet{}'.format(
            args.layers)](pretrained=args.pretrained)
        if not args.pretrained:
            pretrained_model.apply(init_weights)
        #self.maxpool = pretrained_model._modules['maxpool']
        self.conv2 = pretrained_model._modules['layer1']
        self.conv3 = pretrained_model._modules['layer2']
        self.conv4 = pretrained_model._modules['layer3']
        self.conv5 = pretrained_model._modules['layer4']
        del pretrained_model  # clear memory

        # define number of intermediate channels
        if args.layers <= 34:
            num_channels = 512
        elif args.layers >= 50:
            num_channels = 2048
        self.conv6 = conv_bn_relu(num_channels,
                                  512,
                                  kernel_size=3,
                                  stride=2,
                                  padding=1)

        # decoding layers
        kernel_size = 3
        stride = 2
        self.convt5 = convt_bn_relu(in_channels=512,
                                    out_channels=256,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=1,
                                    output_padding=1)
        self.convt4 = convt_bn_relu(in_channels=768,
                                    out_channels=128,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=1,
                                    output_padding=1)
        self.convt3 = convt_bn_relu(in_channels=(256 + 128),
                                    out_channels=64,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=1,
                                    output_padding=1)
        self.convt2 = convt_bn_relu(in_channels=(128 + 64),
                                    out_channels=64,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=1,
                                    output_padding=1)
        self.convt1 = convt_bn_relu(in_channels=128,
                                    out_channels=64,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=1)
        self.convtf = conv_bn_relu(in_channels=128,
                                   out_channels=1,
                                   kernel_size=1,
                                   stride=1,
                                   bn=False,
                                   relu=False)

    def forward(self, x):


        ################### Q-FIT

        # first layer
        if 'd' in self.modality:
            conv1_d_qfit = self.conv1_d_qfit(x['d'])
        if 'rgb' in self.modality:
            conv1_img_qfit = self.conv1_img_qfit(x['rgb'])
        elif 'g' in self.modality:
            conv1_img_qfit = self.conv1_img_qfit(x['g'])

        if self.modality == 'rgbd' or self.modality == 'gd':
            conv1_qfit = torch.cat((conv1_d_qfit, conv1_img_qfit), 1)
        else:
            conv1_qfit = conv1_d_qfit if (self.modality == 'd') else conv1_img_qfit

        conv2_qfit = self.conv2_qfit(conv1_qfit)
        conv3_qfit = self.conv3_qfit(conv2_qfit)
        conv4_qfit = self.conv4_qfit(conv3_qfit)
        conv5_qfit = self.conv5_qfit(conv4_qfit)
        pre_phi = conv5_qfit.squeeze()[:, :-2]
        self.phi = F.softplus(pre_phi)
        ####################

        # output = self.phi_fc1(x)
        # # output = nn.functional.relu(self.phi_fc1(x))
        # output = self.fc1_bn1(output)
        # output = nn.functional.relu(self.phi_fc2(output))
        # output = self.fc2_bn2(output)
        # output = nn.functional.relu(self.phi_fc2b(output))
        # output = self.fc2_bn2b(output)

        # pre_phi = self.phi_fc3(output)

        # phi = F.softplus(phi_parameter.mean(dim=0))
        #phi = F.softplus(pre_phi)  # now the size of phi is mini_batch by input_dim
        #
        # if self.point_estimate:
        #     # S = phi / torch.sum(phi)
        #     # there is a switch vector for each sample
        #     S = phi / torch.sum(phi, dim=1).unsqueeze(dim=1)  # [batch x featnum]
        #     output = x * S



        ##############3

        # phi = F.softplus(self.parameter)

        #if any(torch.isnan(phi)):
        if any(torch.flatten(torch.isnan(self.phi))):
            print("some Phis are NaN")
        # it looks like too large values are making softplus-transformed values very large and returns NaN.
        # this occurs when optimizing with a large step size (or/and with a high momentum value)

        S = self.phi / torch.sum(self.phi)
        #print("phi from model", self.phi[1, -10:])
        #print("S from model", S[1, -10:])


        #print("S argsor: ", np.argsort(S.detach().cpu().numpy(), None)[-10:])
        #print("parameter from model", self.parameter)

        # Slen=len(S)
        # S_expand = S.repeat(x['d'].shape[-1]).reshape(Slen, x['d'].shape[-1])
        # output = x['d'] * S_expand

        # switch mask
        # S_mask_ext = torch.einsum("i, ijk->ijk", [S, self.parameter_mask])
        # print(S_mask_ext[24][308][733])
        # S_mask = torch.max(S_mask_ext, 0)[0]
        # # for i in range(len(self.parameter)):
        # #     print(f"{i}: {len(np.where(S_mask_ind.detach().cpu().numpy() == i)[0])}")
        # print(x['d'].shape)
        # output = x['d'] * S_mask

        #switch many features
        mask = torch.zeros((self.img_height, self.img_width)).to(device)
        for i in range(len(self.bin_ver)-1):
            for j in range(len(self.bin_hor)-1):
                # print(self.bin_hor[i])
                # print(self.bin_hor[i+1])
                # print(self.bin_hor[j])
                # print(self.bin_hor[j+1])
                mask[self.bin_hor[i]:self.bin_hor[i+1], self.bin_hor[j]:self.bin_hor[j+1]]=S[i,j]#self.parameter[i,j]
        output = x['d'] * mask



        # first layer
        if 'd' in self.modality:
            conv1_d = self.conv1_d(output)
        if 'rgb' in self.modality:
            conv1_img = self.conv1_img(x['rgb'])
        elif 'g' in self.modality:
            conv1_img = self.conv1_img(x['g'])

        if self.modality == 'rgbd' or self.modality == 'gd':
            conv1 = torch.cat((conv1_d, conv1_img), 1)
        else:
            conv1 = conv1_d if (self.modality == 'd') else conv1_img

        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)  # batchsize * ? * 176 * 608
        conv4 = self.conv4(conv3)  # batchsize * ? * 88 * 304
        conv5 = self.conv5(conv4)  # batchsize * ? * 44 * 152
        conv6 = self.conv6(conv5)  # batchsize * ? * 22 * 76

        # decoder
        convt5 = self.convt5(conv6)
        y = torch.cat((convt5, conv5), 1)

        convt4 = self.convt4(y)
        y = torch.cat((convt4, conv4), 1)

        convt3 = self.convt3(y)
        y = torch.cat((convt3, conv3), 1)

        convt2 = self.convt2(y)
        y = torch.cat((convt2, conv2), 1)

        convt1 = self.convt1(y)
        y = torch.cat((convt1, conv1), 1)

        y = self.convtf(y)

        if self.training:
            return 100 * y
        else:
            min_distance = 0.9
            return F.relu(
                100 * y - min_distance
            ) + min_distance  # the minimum range of Velodyne is around 3 feet ~= 0.9m


################################################
################################################
################################################




class DepthCompletionNetQLinesNet(nn.Module):
    def __init__(self, args):
        assert (
            args.layers in [18, 34, 50, 101, 152]
        ), 'Only layers 18, 34, 50, 101, and 152 are defined, but got {}'.format(
            layers)
        super(DepthCompletionNetQLinesNet, self).__init__()
        self.modality = args.input

        self.img_height=352
        self.img_width=1216


        self.bin_ver = np.arange(0, self.img_height, 40)
        self.bin_ver = np.append(self.bin_ver, self.img_height)
        self.bin_hor = np.arange(0, self.img_width, 40)
        self.bin_hor = np.append(self.bin_hor, self.img_width)

        num = 352
        num = 65
        #self.parameter = Parameter(-1e-10 * torch.ones(num), requires_grad=True)
        self.parameter = Parameter(-1e-10 * torch.ones(len(self.bin_ver)-1 , len(self.bin_hor)-1))
        self.phi = None
        self.parameter_mask = torch.Tensor(np.load("features/kitti_pixels_to_lines_masks.npy", allow_pickle=True)).to(device)

##################### Q-FIT LINES

        if 'd' in self.modality:
            channels = 64 // len(self.modality)
            #channels = 16
            self.conv1_d_qfit = conv_bn_relu(1,
                                        channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        if 'rgb' in self.modality:
            channels = 64 * 3 // len(self.modality)
            self.conv1_img_qfit = conv_bn_relu(3,
                                          channels,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1)
        elif 'g' in self.modality:
            channels = 64 // len(self.modality)
            self.conv1_img_qfit = conv_bn_relu(1,
                                          channels,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1)

        self.conv2_qfit = conv_bn_relu(64,
                                           channels,
                                           kernel_size=5,
                                           stride=3,
                                           padding=1)

        self.conv3_qfit = conv_bn_relu(channels,
                                       channels,
                                       kernel_size=5,
                                       stride=3,
                                       padding=1)

        self.conv4_qfit = conv_bn_relu(channels,
                                       channels,
                                       kernel_size=5,
                                       stride=2,
                                       padding=1)

        self.conv5_qfit = conv_bn_relu(channels,
                                       1,
                                       kernel_size=5,
                                       stride=2,
                                       padding=1)

        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(297, 65)

        #self.fc1_qfit = lin_bn_relu(-1,65) #297

        # pretrained_model = resnet.__dict__['resnet{}'.format(
        #     args.layers)](pretrained=args.pretrained)
        # if not args.pretrained:
        #     pretrained_model.apply(init_weights)
        # self.maxpool = pretrained_model._modules['maxpool']
        # self.conv2 = pretrained_model._modules['layer1']
        # self.conv3 = pretrained_model._modules['layer2']


#################################3


        if 'd' in self.modality:
            channels = 64 // len(self.modality)
            self.conv1_d = conv_bn_relu(1,
                                        channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        if 'rgb' in self.modality:
            channels = 64 * 3 // len(self.modality)
            self.conv1_img = conv_bn_relu(3,
                                          channels,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1)
        elif 'g' in self.modality:
            channels = 64 // len(self.modality)
            self.conv1_img = conv_bn_relu(1,
                                          channels,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1)

        pretrained_model = resnet.__dict__['resnet{}'.format(
            args.layers)](pretrained=args.pretrained)
        if not args.pretrained:
            pretrained_model.apply(init_weights)
        #self.maxpool = pretrained_model._modules['maxpool']
        self.conv2 = pretrained_model._modules['layer1']
        self.conv3 = pretrained_model._modules['layer2']
        self.conv4 = pretrained_model._modules['layer3']
        self.conv5 = pretrained_model._modules['layer4']
        del pretrained_model  # clear memory

        # define number of intermediate channels
        if args.layers <= 34:
            num_channels = 512
        elif args.layers >= 50:
            num_channels = 2048
        self.conv6 = conv_bn_relu(num_channels,
                                  512,
                                  kernel_size=3,
                                  stride=2,
                                  padding=1)

        # decoding layers
        kernel_size = 3
        stride = 2
        self.convt5 = convt_bn_relu(in_channels=512,
                                    out_channels=256,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=1,
                                    output_padding=1)
        self.convt4 = convt_bn_relu(in_channels=768,
                                    out_channels=128,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=1,
                                    output_padding=1)
        self.convt3 = convt_bn_relu(in_channels=(256 + 128),
                                    out_channels=64,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=1,
                                    output_padding=1)
        self.convt2 = convt_bn_relu(in_channels=(128 + 64),
                                    out_channels=64,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=1,
                                    output_padding=1)
        self.convt1 = convt_bn_relu(in_channels=128,
                                    out_channels=64,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=1)
        self.convtf = conv_bn_relu(in_channels=128,
                                   out_channels=1,
                                   kernel_size=1,
                                   stride=1,
                                   bn=False,
                                   relu=False)

    def forward(self, x):


        ################### Q-FIT

        # first layer
        if 'd' in self.modality:
            conv1_d_qfit = self.conv1_d_qfit(x['d'])
        if 'rgb' in self.modality:
            conv1_img_qfit = self.conv1_img_qfit(x['rgb'])
        elif 'g' in self.modality:
            conv1_img_qfit = self.conv1_img_qfit(x['g'])

        if self.modality == 'rgbd' or self.modality == 'gd':
            conv1_qfit = torch.cat((conv1_d_qfit, conv1_img_qfit), 1)
        else:
            conv1_qfit = conv1_d_qfit if (self.modality == 'd') else conv1_img_qfit

        conv2_qfit = self.conv2_qfit(conv1_qfit)
        conv3_qfit = self.conv3_qfit(conv2_qfit)
        conv4_qfit = self.conv4_qfit(conv3_qfit)
        conv5_qfit = self.conv5_qfit(conv4_qfit)
        flat = self.flatten(conv5_qfit)
        fc1_qfit = self.lin1(flat)
        #fc1_qfit = self.fc1_qfit(flat)
        #pre_phi = conv5_qfit.squeeze()[:, :-2]
        self.phi = F.softplus(fc1_qfit)
        ####################

        if any(torch.flatten(torch.isnan(self.phi))):
            print("some Phis are NaN")
        # it looks like too large values are making softplus-transformed values very large and returns NaN.
        # this occurs when optimizing with a large step size (or/and with a high momentum value)

        S = self.phi / torch.sum(self.phi)
        S = S.squeeze()
        # switch mask
        S_mask_ext = torch.einsum("i, ijk->ijk", [S, self.parameter_mask])
        #print(S_mask_ext[24][308][733])
        print(f"S: {S}")
        S_mask = torch.max(S_mask_ext, 0)[0]

        # for i in range(len(self.parameter)):
        #     print(f"{i}: {len(np.where(S_mask_ind.detach().cpu().numpy() == i)[0])}")
        output = x['d'] * S_mask

        #
        # #switch many features
        # mask = torch.zeros((self.img_height, self.img_width)).to(device)
        # for i in range(len(self.bin_ver)-1):
        #     for j in range(len(self.bin_hor)-1):
        #         # print(self.bin_hor[i])
        #         # print(self.bin_hor[i+1])
        #         # print(self.bin_hor[j])
        #         # print(self.bin_hor[j+1])
        #         mask[self.bin_hor[i]:self.bin_hor[i+1], self.bin_hor[j]:self.bin_hor[j+1]]=S[i,j]#self.parameter[i,j]
        # output = x['d'] * mask



        # first layer
        if 'd' in self.modality:
            conv1_d = self.conv1_d(output)
        if 'rgb' in self.modality:
            conv1_img = self.conv1_img(x['rgb'])
        elif 'g' in self.modality:
            conv1_img = self.conv1_img(x['g'])

        if self.modality == 'rgbd' or self.modality == 'gd':
            conv1 = torch.cat((conv1_d, conv1_img), 1)
        else:
            conv1 = conv1_d if (self.modality == 'd') else conv1_img

        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)  # batchsize * ? * 176 * 608
        conv4 = self.conv4(conv3)  # batchsize * ? * 88 * 304
        conv5 = self.conv5(conv4)  # batchsize * ? * 44 * 152
        conv6 = self.conv6(conv5)  # batchsize * ? * 22 * 76

        # decoder
        convt5 = self.convt5(conv6)
        y = torch.cat((convt5, conv5), 1)

        convt4 = self.convt4(y)
        y = torch.cat((convt4, conv4), 1)

        convt3 = self.convt3(y)
        y = torch.cat((convt3, conv3), 1)

        convt2 = self.convt2(y)
        y = torch.cat((convt2, conv2), 1)

        convt1 = self.convt1(y)
        y = torch.cat((convt1, conv1), 1)

        y = self.convtf(y)

        if self.training:
            return 100 * y
        else:
            min_distance = 0.9
            return F.relu(
                100 * y - min_distance
            ) + min_distance  # the minimum range of Velodyne is around 3 feet ~= 0.9m
