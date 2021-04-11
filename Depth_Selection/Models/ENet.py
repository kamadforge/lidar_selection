'''
A quick, partial implementation of ENet (https://arxiv.org/abs/1606.02147) using PyTorch.
The original Torch ENet implementation can process a 480x360 image in ~12 ms (on a P2 AWS
instance).  TensorFlow takes ~35 ms.  The PyTorch implementation takes ~25 ms, an improvement
over TensorFlow, but worse than the original Torch.
'''

from __future__ import absolute_import


import sys
import time


import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import functools
import pdb

def define_params(no_dropout):
    dropout = (not no_dropout)*0.1
    ENCODER_PARAMS = [
            {
                'internal_scale': 4,
                'use_relu': True,
                'asymmetric': False,
                'dilated': False,
                'input_channels': 16,
                'output_channels': 64,
                'downsample': True,
                'dropout_prob': dropout/10
            },
            {
                'internal_scale': 4,
                'use_relu': True,
                'asymmetric': False,
                'dilated': False,
                'input_channels': 64,
                'output_channels': 64,
                'downsample': False,
                'dropout_prob': dropout/10
            },
            {
                'internal_scale': 4,
                'use_relu': True,
                'asymmetric': False,
                'dilated': False,
                'input_channels': 64,
                'output_channels': 64,
                'downsample': False,
                'dropout_prob': dropout/10
            },
            {
                'internal_scale': 4,
                'use_relu': True,
                'asymmetric': False,
                'dilated': False,
                'input_channels': 64,
                'output_channels': 64,
                'downsample': False,
                'dropout_prob': dropout/10
            },
            {
                'internal_scale': 4,
                'use_relu': True,
                'asymmetric': False,
                'dilated': False,
                'input_channels': 64,
                'output_channels': 64,
                'downsample': False,
                'dropout_prob': dropout/10
            },
            {
                'internal_scale': 4,
                'use_relu': True,
                'asymmetric': False,
                'dilated': False,
                'input_channels': 64,
                'output_channels': 128,
                'downsample': True,
                'dropout_prob': dropout
            },
            {
                'internal_scale': 4,
                'use_relu': True,
                'asymmetric': False,
                'dilated': False,
                'input_channels': 128,
                'output_channels': 128,
                'downsample': False,
                'dropout_prob': dropout
            },
            {
                'internal_scale': 4,
                'use_relu': True,
                'asymmetric': False,
                'dilated': 2,
                'input_channels': 128,
                'output_channels': 128,
                'downsample': False,
                'dropout_prob': dropout
            },
            {
                'internal_scale': 4,
                'use_relu': True,
                'asymmetric': 5,
                'dilated': False,
                'input_channels': 128,
                'output_channels': 128,
                'downsample': False,
                'dropout_prob': dropout
            },
            {
                'internal_scale': 4,
                'use_relu': True,
                'asymmetric': False,
                'dilated': 4,
                'input_channels': 128,
                'output_channels': 128,
                'downsample': False,
                'dropout_prob': dropout
            },
            {
                'internal_scale': 4,
                'use_relu': True,
                'asymmetric': False,
                'dilated': False,
                'input_channels': 128,
                'output_channels': 128,
                'downsample': False,
                'dropout_prob': dropout
            },
            {
                'internal_scale': 4,
                'use_relu': True,
                'asymmetric': False,
                'dilated': 8,
                'input_channels': 128,
                'output_channels': 128,
                'downsample': False,
                'dropout_prob': dropout
            },
            {
                'internal_scale': 4,
                'use_relu': True,
                'asymmetric': False,
                'dilated': False,
                'input_channels': 128,
                'output_channels': 128,
                'downsample': False,
                'dropout_prob': dropout
            },
            {
                'internal_scale': 4,
                'use_relu': True,
                'asymmetric': 5,
                'dilated': False,
                'input_channels': 128,
                'output_channels': 128,
                'downsample': False,
                'dropout_prob': dropout
            },
            {
                'internal_scale': 4,
                'use_relu': True,
                'asymmetric': False,
                'dilated': 16,
                'input_channels': 128,
                'output_channels': 128,
                'downsample': False,
                'dropout_prob': dropout
            },  
            
                
            {   
                'internal_scale': 4,
                'use_relu': True,
                'asymmetric': False,
                'dilated': False,
                'input_channels': 128,
                'output_channels': 128,
                'downsample': False,
                'dropout_prob': dropout
            },
            
               
            {
                'internal_scale': 4,
                'use_relu': True,
                'asymmetric': False,
                'dilated': 2,
                'input_channels': 128,
                'output_channels': 128,
                'downsample': False,
                'dropout_prob': dropout
            },
            {
                'internal_scale': 4,
                'use_relu': True,
                'asymmetric': 5,
                'dilated': False,
                'input_channels': 128,
                'output_channels': 128,
                'downsample': False,
                'dropout_prob': dropout
            },
            {
                'internal_scale': 4,
                'use_relu': True,
                'asymmetric': False,
                'dilated': 4,
                'input_channels': 128,
                'output_channels': 128,
                'downsample': False,
                'dropout_prob': dropout
            },
            {
                'internal_scale': 4,
                'use_relu': True,
                'asymmetric': False,
                'dilated': False,
                'input_channels': 128,
                'output_channels': 128,
                'downsample': False,
                'dropout_prob': dropout
            },
            {
                'internal_scale': 4,
                'use_relu': True,
                'asymmetric': False,
                'dilated': 8,
                'input_channels': 128,
                'output_channels': 128,
                'downsample': False,
                'dropout_prob': dropout
            },
            {
                'internal_scale': 4,
                'use_relu': True,
                'asymmetric': 5,
                'dilated': False,
                'input_channels': 128,
                'output_channels': 128,
                'downsample': False,
                'dropout_prob': dropout
            },
            {
                'internal_scale': 4,
                'use_relu': True,
                'asymmetric': False,
                'dilated': 16,
                'input_channels': 128,
                'output_channels': 128,
                'downsample': False,
                'dropout_prob': dropout
            }
        ]
    
    
    DECODER_PARAMS = [
            {
                'input_channels': 128,
                'output_channels': 128,
                'upsample': False,
                'pooling_module': None
            },
            {
                'input_channels': 128,
                'output_channels': 64,
                'upsample': True,
                'pooling_module': None
            },
            {
                'input_channels': 64,
                'output_channels': 64,
                'upsample': False,
                'pooling_module': None
            },
            {
                'input_channels': 64,
                'output_channels': 64,
                'upsample': False,
                'pooling_module': None
            },
            {
                'input_channels': 64,
                'output_channels': 16,
                'upsample': True,
                'pooling_module': None
            },
            {
                'input_channels': 16,
                'output_channels': 16,
                'upsample': False,
                'pooling_module': None
            }
        ]
    return ENCODER_PARAMS, DECODER_PARAMS


def norm_layer(norm_type='batch'):
    if norm_type == 'batch' or 'none':
        norm_layer = functools.partial(nn.BatchNorm2d, eps=1e-5, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('norm type is not implemented'.format(norm_type))
    return norm_layer

class InitialBlock(nn.Module):
    def __init__(self, inchannels, norm_layer):
        super().__init__()
        self.conv = nn.Conv2d(
            inchannels, 16-inchannels, (3, 3),
            stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.batch_norm = norm_layer(16)
        self.prelu = nn.PReLU(16)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.batch_norm(output)
        return self.prelu(output)


class EncoderMainPath(nn.Module):
    def __init__(self,norm_layer, internal_scale=None, use_relu=None, asymmetric=None, dilated=None, input_channels=None, output_channels=None, downsample=None, dropout_prob=None):
        super().__init__()

        internal_channels = output_channels // internal_scale
        input_stride = downsample and 2 or 1

        self.__dict__.update(locals())
        del self.self

        self.input_conv = nn.Conv2d(
            input_channels, internal_channels, input_stride,
            stride=input_stride, padding=0, bias=False)

        self.input_batch_norm = norm_layer(internal_channels)
        
        self.prelu = nn.PReLU(internal_channels)

        if not dilated and not asymmetric:
            self.middle_conv = nn.Conv2d(
                internal_channels, internal_channels, 3,
                stride=1, padding=1, bias=True)
        
        elif asymmetric:
            pad = (self.asymmetric-1)//2
            self.asymmetric1 = nn.Conv2d(
                internal_channels, internal_channels, (self.asymmetric,1),
                stride=1, padding=(pad,0), bias=False)
            self.asymmetric2 = nn.Conv2d(
                internal_channels, internal_channels, (1,self.asymmetric),
                stride=1, padding=(0,pad), bias=True)
        elif dilated: 
            self.dilated = nn.Conv2d(
                internal_channels, internal_channels, 3,
                stride=1, padding=dilated, dilation = dilated, bias=True)
         
        self.middle_batch_norm = norm_layer(internal_channels)

        self.output_conv = nn.Conv2d(
            internal_channels, output_channels, 1,
            stride=1, padding=0, bias=False)

        self.output_batch_norm = norm_layer(output_channels)

        self.dropout = nn.Dropout2d(dropout_prob)


    def forward(self, input):
        output = self.input_conv(input)
        
        output = self.input_batch_norm(output)

        output = self.prelu(output)
        
        if not self.dilated and not self.asymmetric:
            output = self.middle_conv(output)
        elif self.asymmetric:
            output = self.asymmetric1(output)
            output = self.asymmetric2(output)
        elif self.dilated:
            output = self.dilated(output)
            
        output = self.middle_batch_norm(output)

        output = self.prelu(output)

        output = self.output_conv(output)

        output = self.output_batch_norm(output)

        output = self.dropout(output)
        
        return output


class EncoderOtherPath(nn.Module):
    def __init__(self, internal_scale=None, use_relu=None, asymmetric=None, dilated=None, input_channels=None, output_channels=None, downsample=None, **kwargs):
        super().__init__()

        self.__dict__.update(locals())
        del self.self

        if downsample:
            self.pool = nn.MaxPool2d(2, stride=2, return_indices=True)

    def forward(self, input):
        output = input

        if self.downsample:
            output, self.indices = self.pool(input)

        if self.output_channels != self.input_channels:
            new_size = [1, 1, 1, 1]
            new_size[1] = self.output_channels // self.input_channels
            output = output.repeat(*new_size)

        return output


class EncoderModule(nn.Module):
    def __init__(self, norm_layer, **kwargs):
        super().__init__()
        self.main = EncoderMainPath(norm_layer, **kwargs)
        self.other = EncoderOtherPath(**kwargs)
        self.prelu = nn.PReLU(self.other.output_channels)

    def forward(self, input):
        main = self.main(input)
        other = self.other(input)
        return self.prelu(main + other)


class Encoder(nn.Module):
    def __init__(self, params, inchannels, nclasses, norm_layer):
        super().__init__()
        self.initial_block = InitialBlock(inchannels, norm_layer)

        self.layers = []
        for i, params in enumerate(params):
            layer_name = 'encoder_{:02d}'.format(i)
            layer = EncoderModule(norm_layer, **params)
            super().__setattr__(layer_name, layer)
            self.layers.append(layer)

        self.output_conv = nn.Conv2d(
            128, nclasses, 1,
            stride=1, padding=0, bias=True)
        self.fully_connected1 = nn.Linear(12800, nclasses)

    def forward(self, input, predict=False):
        output = self.initial_block(input)

        for i, layer in enumerate(self.layers):
            output = layer(output)

        if predict:
            output = F.relu(self.output_conv(output))
            output = output.view(output.size()[0],-1)
            output = self.fully_connected1(output)
        return output


class DecoderMainPath(nn.Module):
    def __init__(self, norm_layer, input_channels=None, output_channels=None, upsample=None, pooling_module=None):
        super().__init__()

        internal_channels = output_channels // 4
        input_stride = 2 if upsample is True else 1

        self.__dict__.update(locals())
        del self.self

        self.input_conv = nn.Conv2d(
            input_channels, internal_channels, 1,
            stride=1, padding=0, bias=False)

        self.input_batch_norm = norm_layer(internal_channels)

        if not upsample:
            self.middle_conv = nn.Conv2d(
                internal_channels, internal_channels, 3,
                stride=1, padding=1, bias=True)
        else:
            self.middle_conv = nn.ConvTranspose2d(
                internal_channels, internal_channels, 3,
                stride=2, padding=1, output_padding=1,
                bias=True)

        self.middle_batch_norm = norm_layer(internal_channels)

        self.output_conv = nn.Conv2d(
            internal_channels, output_channels, 1,
            stride=1, padding=0, bias=False)

        self.output_batch_norm = norm_layer(output_channels)

    def forward(self, input):
        output = self.input_conv(input)

        output = self.input_batch_norm(output)

        output = F.relu(output)

        output = self.middle_conv(output)

        output = self.middle_batch_norm(output)

        output = F.relu(output)

        output = self.output_conv(output)

        output = self.output_batch_norm(output)

        return output


class DecoderOtherPath(nn.Module):
    def __init__(self, norm_layer, input_channels=None, output_channels=None, upsample=None, pooling_module=None):
        super().__init__()

        self.__dict__.update(locals())
        del self.self

        if output_channels != input_channels or upsample:
            self.conv = nn.Conv2d(
                input_channels, output_channels, 1,
                stride=1, padding=0, bias=False)
            self.batch_norm = norm_layer(output_channels)
            if upsample and pooling_module:
                self.unpool = nn.MaxUnpool2d(2, stride=2, padding=0)

    def forward(self, input):
        output = input

        if self.output_channels != self.input_channels or self.upsample:
            output = self.conv(output)
            output = self.batch_norm(output)
            if self.upsample and self.pooling_module:
                output_size = list(output.size())
                output_size[2] *= 2
                output_size[3] *= 2
                output = self.unpool(
                    output, self.pooling_module.indices,
                    output_size=output_size)

        return output


class DecoderModule(nn.Module):
    def __init__(self,norm_layer, **kwargs):
        super().__init__()

        self.main = DecoderMainPath(norm_layer, **kwargs)
        self.other = DecoderOtherPath(norm_layer, **kwargs)
                                                    
    def forward(self, input):
        main = self.main(input)
        other = self.other(input)
        return F.relu(main + other)


class Decoder(nn.Module):
    def __init__(self, params, nclasses, encoder, norm_layer):
        super().__init__()

        self.encoder = encoder

        self.pooling_modules = []

        for mod in self.encoder.modules():
            try:
                if mod.other.downsample:
                    self.pooling_modules.append(mod.other)
            except AttributeError:
                pass

        self.layers = []
        for i, params in enumerate(params):
            if params['upsample']:
                params['pooling_module'] = self.pooling_modules.pop(-1)
                
            layer = DecoderModule(norm_layer, **params)
            self.layers.append(layer)
            layer_name = 'decoder{:02d}'.format(i)
            super().__setattr__(layer_name, layer)

        self.output_conv = nn.ConvTranspose2d(
                16, nclasses, 2,
                stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = self.encoder(input, predict=False)
        output_encoder = output
        
        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output_encoder, output

class ENet(nn.Module):
    def __init__(self, args, nclasses=1, norm='batch', no_dropout = False):
        super().__init__()
        self.norm_layer = norm_layer(norm)

        ENCODER_PARAMS, DECODER_PARAMS = define_params(no_dropout)
        self.encoder = Encoder(ENCODER_PARAMS, args.channels_in, nclasses, self.norm_layer)
        self.decoder = Decoder(DECODER_PARAMS, nclasses, self.encoder, self.norm_layer)
        
    def forward(self, input, only_encode=False, predict=False):
        if only_encode and predict:
            return self.encoder.forward(input, predict=predict)
        
        else:
            output_encoder, seg_output = self.decoder.forward(input)
            return seg_output
