#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .model import Model
from .model_test import Model as test
from .sparse_to_dense import DepthCompletionNet
from .DispNet import DispNetS
from .DepthRecNet import DepthRecNet
from .ERFNet_stacked import Net as ERFNet_stacked
from .ERFNet_single import Net as ERFNet_single
from .local import Model as Local_net
from .stacked_resnet import DepthCompletionNet as double_res
from .refinenet import Model as stacked_refine
from .resnet import ResNet_EncDec
from .deeplab import DeepLabv3_plus

model_dict = {'mod': Model, 'sparse': DepthCompletionNet, 'dispnet': DispNetS, 'recurrent': DepthRecNet, 'stacked_erfnet': ERFNet_stacked, 'erfnet': ERFNet_single, 'local': Local_net, 'double_res': double_res, 'stacked_refine': stacked_refine, 'mod_test': test, 'resnet': ResNet_EncDec, 'deeplab': DeepLabv3_plus}

def allowed_models():
    return model_dict.keys()


def define_model(mod, args):
    if mod not in allowed_models():
        raise KeyError("The requested model: {} is not implemented".format(mod))
    else:
        return model_dict[mod](args)
