import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet, ResNet
import pdb


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


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
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride,
        padding, bias=bias))
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
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
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
        stride, padding, output_padding, bias=bias))
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    layers = nn.Sequential(*layers)

    # initialize the weights
    for m in layers.modules():
        init_weights(m)

    return layers

class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if downsample is not None:
            stride=2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class DepthCompletionNet(nn.Module):
    def __init__(self, args):
        assert (args.layers in [18, 34, 50, 101, 152]), 'Only layers 18, 34, 50, 101, and 152 are defined, but got {}'.format(layers)
        super(DepthCompletionNet, self).__init__()
        # define number of intermediate channels
        if args.layers <= 34:
            num_channels = 512
        elif args.layers >= 50:
            num_channels = 2048
        self.norm_layer = nn.BatchNorm2d

        self.rgb_layer1 = conv_bn_relu(3, 32, kernel_size=3, stride=1, padding=1)
        self.rgb_layer2 = self._make_layer(32, 32, 2, downsample=None)
        self.rgb_layer3 = self._make_layer(32, 64, 2, downsample=1)
        self.rgb_layer4 = self._make_layer(64, 128, 2, downsample=1)
        self.rgb_layer5 = self._make_layer(128, 256, 2, downsample=1)
        self.rgb_layer6 = conv_bn_relu(256, 256, kernel_size=3, stride=2, padding=1)

        self.depth_layer1 = conv_bn_relu(1, 32, kernel_size=3, stride=1, padding=1)
        self.depth_layer2 = self._make_layer(32, 32, 2, downsample=None)
        self.depth_layer3 = self._make_layer(64, 64, 2, downsample=1)
        self.depth_layer4 = self._make_layer(128, 128, 2, downsample=1)
        self.depth_layer5 = self._make_layer(256, 256, 2, downsample=1)
        self.depth_layer6 = conv_bn_relu(512, 256, kernel_size=3, stride=2, padding=1)

        # decoding layers 1
        kernel_size = 3
        stride = 2
        self.convt5 = convt_bn_relu(in_channels=256, out_channels=256,
            kernel_size=kernel_size, stride=stride, padding=1, output_padding=1)
        self.convt4 = convt_bn_relu(in_channels=256, out_channels=128,
            kernel_size=kernel_size, stride=stride, padding=1, output_padding=1)
        self.convt3 = convt_bn_relu(in_channels=128, out_channels=64,
            kernel_size=kernel_size, stride=stride, padding=1, output_padding=1)
        self.convt2 = convt_bn_relu(in_channels=64, out_channels=32,
            kernel_size=kernel_size, stride=stride, padding=1, output_padding=1)
        self.convt1 = convt_bn_relu(in_channels=32, out_channels=32,
            kernel_size=kernel_size, stride=1, padding=1)
        # self.convtf = conv_bn_relu(in_channels=64, out_channels=1, kernel_size=1, stride=1, bn=False, relu=False)

        # decoding layers 2
        kernel_size = 3
        stride = 2
        self.convt5_d= convt_bn_relu(in_channels=256, out_channels=256,
            kernel_size=kernel_size, stride=stride, padding=1, output_padding=1)
        self.convt4_d= convt_bn_relu(in_channels=256, out_channels=128,
            kernel_size=kernel_size, stride=stride, padding=1, output_padding=1)
        self.convt3_d = convt_bn_relu(in_channels=(128), out_channels=64,
            kernel_size=kernel_size, stride=stride, padding=1, output_padding=1)
        self.convt2_d = convt_bn_relu(in_channels=(64), out_channels=32,
            kernel_size=kernel_size, stride=stride, padding=1, output_padding=1)
        self.convt1_d = convt_bn_relu(in_channels=32, out_channels=32,
            kernel_size=kernel_size, stride=1, padding=1)
        self.convtf = conv_bn_relu(in_channels=32, out_channels=1, kernel_size=1, stride=1, bn=False, relu=False)


    def _make_layer(self, inchannels, outchannels, blocks, downsample):
            layers = []
            if downsample:
                downsample = nn.Sequential(conv1x1(inchannels, outchannels, stride=2) , self.norm_layer(outchannels))
            layers.append(BasicBlock(inplanes=inchannels, planes=outchannels, downsample=downsample))
            for _ in range(1, blocks):
                layers.append(BasicBlock(outchannels, outchannels))

            return nn.Sequential(*layers)

    def forward(self, input):
        # first layer

        rgb_in = input[:, 1:, :, :]
        lidar_in = input[:, 0:1, :, :]

        # Encoder rgb
        conv1 = self.rgb_layer1(rgb_in)
        conv2 = self.rgb_layer2(conv1) # batchsize * 64  * 256 * 1216
        conv3 = self.rgb_layer3(conv2) # batchsize * 128 * 128 * 608
        conv4 = self.rgb_layer4(conv3) # batchsize * 256 * 64  * 304
        conv5 = self.rgb_layer5(conv4) # batchsize * 512 * 32  * 152
        conv6 = self.rgb_layer6(conv5) # batchsize * 512 * 16  * 76

        # Decoder rgb
        convt5 = self.convt5(conv6)
        y5 = convt5 + conv5
        convt4 = self.convt4(y5)
        y4 = convt4 + conv4
        convt3 = self.convt3(y4)
        y3 = convt3 + conv3
        convt2 = self.convt2(y3)
        y2 = convt2 + conv2
        # convt1 = self.convt1(y)

        # Encoder lidar
        conv1_d = self.depth_layer1(lidar_in)

        conv2_d = self.depth_layer2(conv1_d) # batchsize * 64  * 256 * 1216
        y2 = torch.cat((conv2_d, y2), 1)

        conv3_d = self.depth_layer3(y2) # batchsize * 128 * 128 * 608
        y3 = torch.cat((convt3, y3), 1)

        conv4_d = self.depth_layer4(y3) # batchsize * 256 * 64  * 304
        y4 = torch.cat((convt4, y4), 1)

        conv5_d = self.depth_layer5(y4) # batchsize * 512 * 32  * 152
        y5 = torch.cat((convt5, y5), 1)

        conv6_d = self.depth_layer6(y5) # batchsize * 512 * 16  * 76

        # Decoder rgb
        y6 = conv6_d + conv6
        convt5 = self.convt5_d(y6)
        y5 = convt5 + conv5_d
        convt4 = self.convt4(y5)
        y4 = convt4 + conv4_d
        convt3 = self.convt3(y4)
        y3 = convt3 + conv3_d
        convt2 = self.convt2(y3)
        y2 = convt2 + conv2_d
        convt1 = self.convt1(y2)
        y = self.convtf(convt1)

        return y 
