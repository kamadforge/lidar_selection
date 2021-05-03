#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import torch
import torch.nn as nn
import torchvision.models
import torch.nn.functional as F
import collections

# Init Gaussian random weights
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

class Unpool(nn.Module):
    def __init__(self, num_channels, stride=2):
        super(Unpool, self).__init__()

        self.num_channels = num_channels
        self.stride = stride

        # create kernel [1, 0; 0, 0]
        self.weights = torch.autograd.Variable(
            torch.zeros(num_channels, 1, stride, stride).cuda())
        self.weights[:, :, 0, 0] = 1

    def forward(self, x):
        return F.conv_transpose2d(x, self.weights, stride=self.stride, groups=self.num_channels)


class UpProjModule(nn.Module):
    def __init__(self, in_channels):
        super(UpProjModule, self).__init__()
        out_channels = in_channels // 2
        self.unpool = Unpool(in_channels)
        self.upper_branch = nn.Sequential(collections.OrderedDict([
            ('conv1', nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)),
            ('batchnorm1', nn.BatchNorm2d(out_channels)),
            ('relu', nn.ReLU()),
            ('conv2', nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)),
            ('batchnorm2', nn.BatchNorm2d(out_channels)),
        ]))
        self.bottom_branch = nn.Sequential(collections.OrderedDict([
            ('conv', nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)),
            ('batchnorm', nn.BatchNorm2d(out_channels)),
        ]))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.unpool(x)
        x1 = self.upper_branch(x)
        x2 = self.bottom_branch(x)
        x = x1 + x2
        x = self.relu(x)
        return x


class DepthRecNet(nn.Module):
    def __init__(self, args):

        super(DepthRecNet, self).__init__()
        num_channels = 512

        self.in_channels = args.channels_in

        ########################
        if self.in_channels == 4:
            self.conv1_img = nn.Conv2d(3, 32, kernel_size=3, stride=4, padding=3, bias=False)
            self.bn1_img = nn.BatchNorm2d(32)

            self.conv1_d = nn.Conv2d(1, 32, kernel_size=7, stride=4, padding=4, bias=False)
            self.bn1_d = nn.BatchNorm2d(32)

            self.conv1_img.apply(weights_init)
            self.bn1_img.apply(weights_init)
            self.conv1_d.apply(weights_init)
            self.bn1_d.apply(weights_init)
        else:
            self.conv1_img = nn.Conv2d(3, 64, kernel_size=3, stride=4, padding=3, bias=False)
            self.bn1_img = nn.BatchNorm2d(64)

            self.conv1_img.apply(weights_init)
            self.bn1_img.apply(weights_init)


        ############ Resnet ###############
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(args.layers)](pretrained=args.pretrained)
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']

        # clear memory
        del pretrained_model

        ############## Conv LSTM ###########

        self.conv_i = nn.Sequential(
            nn.Conv2d(num_channels * 2, num_channels, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(num_channels * 2, num_channels, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(num_channels * 2, num_channels, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(num_channels * 2, num_channels, 3, 1, 1),
            nn.Sigmoid()
            )

        self.conv_i.apply(weights_init)
        self.conv_f.apply(weights_init)
        self.conv_g.apply(weights_init)
        self.conv_o.apply(weights_init)

        ######################################

        self.conv2 = nn.Conv2d(num_channels, num_channels // 2, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels // 2)

        self.conv2.apply(weights_init)
        self.bn2.apply(weights_init)

        ############## Decoder ##############

        self.decoder_layer1 = UpProjModule(num_channels // 2)
        self.decoder_layer2 = UpProjModule(num_channels // 4)
        self.decoder_layer3 = UpProjModule(num_channels // 8)
        self.decoder_layer4 = UpProjModule(num_channels // 16)

        self.decoder_layer1.apply(weights_init)
        self.decoder_layer2.apply(weights_init)
        self.decoder_layer3.apply(weights_init)
        self.decoder_layer4.apply(weights_init)

        ############### Up Sampling ##########

        self.conv3 = nn.Conv2d(num_channels // 32, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3.apply(weights_init)

    def forward(self, x, h=None, c=None, training=True):

        # img = x[:, 0:3, :, :]
        # depth = (x[:, 3, :, :]).unsqueeze(1)

        img = x[:, 1:, :, :]

        x_img = self.conv1_img(img)
        x_img = self.bn1_img(x_img)
        x_img = self.relu(x_img)

        if self.in_channels == 4:
            depth = x[:, 0:1, :, :]
            x_d = self.conv1_d(depth)
            x_d = self.bn1_d(x_d)
            x_d = self.relu(x_d)
            x = torch.cat((x_d, x_img), 1)
        else:
            x = x_img

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x_lstm = torch.cat((x, h), 1)
        # i = self.conv_i(x_lstm)
        # f = self.conv_f(x_lstm)
        # g = self.conv_g(x_lstm)
        # o = self.conv_o(x_lstm)
        # c = f * c + i * g
        # h = o * F.tanh(c)
        # x = self.conv2(h)

        x = self.conv2(x)

        x = self.bn2(x)

        x = self.decoder_layer1(x)  # B, 64, 14, 56
        x = self.decoder_layer2(x)  # B, 64, 28, 112
        x = self.decoder_layer3(x)  # B, 32, 56, 224
        x = self.decoder_layer4(x)  # B, 16, 112, 448

        x = self.conv3(x)
        x = nn.functional.interpolate(x, size=(img.size()[2], img.size()[3]), mode='bilinear', align_corners=False)

        return x #, h, c
