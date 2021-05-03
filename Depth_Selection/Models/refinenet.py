import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from .ERFNet import Net as ERFNet
from .ENet import ENet 
from Utils.utils import define_init_weights
from torchvision.models import resnet
import pdb


def conv_bn_relu(in_channels, out_channels, kernel_size, \
        stride=1, padding=0, bn=True, relu=True):
    bias = not bn
    layers = []
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride,
        padding, bias=bias))
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    layers = nn.Sequential(*layers)

    return layers

def convt_bn_relu(in_channels, out_channels, kernel_size, \
        stride=1, padding=0, output_padding=0, bn=True, relu=True):
    bias = not bn
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
        stride, padding, output_padding, bias=bias))
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    layers = nn.Sequential(*layers)

    return layers

class ResNet_custom(nn.Module):
    def __init__(self, args):
        super(ResNet_custom, self).__init__()
        pretrained_model = resnet.__dict__['resnet{}'.format(args.layers)](pretrained=args.pretrained)
        self.conv2 = pretrained_model._modules['layer1']
        self.conv3 = pretrained_model._modules['layer2']
        self.conv4 = pretrained_model._modules['layer3']
        self.conv5 = pretrained_model._modules['layer4']
        del pretrained_model # clear memory

        # define number of intermediate channels
        if args.layers <= 34:
            num_channels = 512
        elif args.layers >= 50:
            num_channels = 2048
        self.conv6 = conv_bn_relu(num_channels, 512, kernel_size=3, stride=2, padding=1)

        # Decoder 
        kernel_size = 3
        stride = 2
        self.convt5 = convt_bn_relu(in_channels=512, out_channels=256,
            kernel_size=kernel_size, stride=stride, padding=1, output_padding=1)
        self.convt4 = convt_bn_relu(in_channels=768, out_channels=128,
            kernel_size=kernel_size, stride=stride, padding=1, output_padding=1)
        self.convt3 = convt_bn_relu(in_channels=(256+128), out_channels=64,
            kernel_size=kernel_size, stride=stride, padding=1, output_padding=1)
        self.convt2 = convt_bn_relu(in_channels=(128+64), out_channels=64,
            kernel_size=kernel_size, stride=stride, padding=1, output_padding=1)
        self.convt1 = convt_bn_relu(in_channels=128, out_channels=64,
            kernel_size=kernel_size, stride=1, padding=1)
        self.convtf = conv_bn_relu(in_channels=128, out_channels=1, kernel_size=1, stride=1, bn=False, relu=False)

    def forward(self, x):
        # 2.a Fusion: encoder
        conv2 = self.conv2(x)
        conv3 = self.conv3(conv2) 
        conv4 = self.conv4(conv3) 
        conv5 = self.conv5(conv4) 
        conv6 = self.conv6(conv5) 

        # 2.b Fusion: decoder
        convt5 = self.convt5(conv6)
        y = torch.cat((convt5, conv5), 1)

        convt4 = self.convt4(y)
        y = torch.cat((convt4, conv4), 1)

        convt3 = self.convt3(y)
        y = torch.cat((convt3, conv3), 1)

        convt2 = self.convt2(y)
        y = torch.cat((convt2, conv2), 1)

        convt1 = self.convt1(y)
        y = torch.cat((convt1,x), 1)

        y = self.convtf(y)
        y = F.interpolate(y, size=(256, 1216), mode='bilinear', align_corners=False)
        return y

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.in_channels = args.channels_in

        #############################
        #    1.Feature Mapping
        #############################

        channels = 32
        stride_new=2
        if args.submod != 'resnet':
            channels=3
            stride_new=1
        self.feature_extractor_depth = nn.Sequential(
                        conv_bn_relu(1, channels, kernel_size=3, stride=1, padding=1),
                        conv_bn_relu(channels, channels, kernel_size=3, stride=stride_new, padding=1))

        self.feature_extractor_rgb = nn.Sequential(
                        conv_bn_relu(3, channels, kernel_size=3, stride=1, padding=1),
                        conv_bn_relu(channels, channels, kernel_size=3, stride=stride_new, padding=1))

        #############################
        #        2.FusionNet
        #############################
        if args.submod == 'resnet':
            self.fusion = ResNet_custom(args)
        elif args.submod == 'erfnet':
            args.channels_in = 6
            self.fusion = ERFNet(args, out_channels=1, drop=0.3, multi=False)
        elif args.submod == 'enet':
            args.channels_in = 6
            self.fusion = ENet(args)
        else:
            assert False

        #############################
        #        3.RefineNet
        #############################

        self.convbnrelu = nn.Sequential(convbn(2, 32, 3, 1, 1, 1),
                                        nn.ReLU(inplace=True))
        self.hourglass1 = hourglass(32)
        self.hourglass2 = hourglass(32)
        self.fuse = nn.Sequential(convbn(32, 32, 3, 1, 1, 1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(32, 1, kernel_size=3, padding=1, stride=1, bias=True))

    def forward(self, input):
        rgb_in = input[:, 1:, :, :]
        lidar_in = input[:, 0:1, :, :]

        # 1. Extract
        conv1_d = self.feature_extractor_depth(lidar_in)
        conv1_img = self.feature_extractor_rgb(rgb_in)

        # Fuse
        conv1 = torch.cat((conv1_d, conv1_img), 1)
        y = self.fusion(conv1)

        # 3. Concat
        input = torch.cat((lidar_in, y), 1)

        # 4. RefineNET
        out = self.convbnrelu(input)
        out1, embedding3, embedding4 = self.hourglass1(out, None, None)
        out1 = out1 + out
        out2, _, _ = self.hourglass2(out1, embedding3, embedding4)
        out2 = out2 + out
        out = self.fuse(out2)

        return out, y 


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_planes))


class hourglass(nn.Module):
    def __init__(self, channels_in):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn(channels_in, channels_in*2, kernel_size=3, stride=2, pad=1, dilation=1),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(channels_in*2, channels_in*2, kernel_size=3, stride=1, pad=1, dilation=1)

        self.conv3 = nn.Sequential(convbn(channels_in*2, channels_in*2, kernel_size=3, stride=2, pad=1, dilation=1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn(channels_in*2, channels_in*4, kernel_size=3, stride=1, pad=1, dilation=1))

        self.conv5 = nn.Sequential(nn.ConvTranspose2d(channels_in*4, channels_in*2, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm2d(channels_in*2),
                                   nn.ReLU(inplace=True))

        self.conv6 = nn.Sequential(nn.ConvTranspose2d(channels_in*2, channels_in, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm2d(channels_in))

    def forward(self, x, em1, em2):
        x = self.conv1(x)
        x = self.conv2(x)
        if em1 is not None:
            x = x + em1
        x = F.relu(x, inplace=True)

        x_prime = self.conv3(x)
        x_prime = self.conv4(x_prime)
        if em2 is not None:
            x_prime = x_prime + em2
        x_prime = F.relu(x_prime, inplace=True)

        out = self.conv5(x_prime)
        out = self.conv6(out)

        return out, x, x_prime
