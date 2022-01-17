# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import CONV_LAYERS
from torch import nn

CONV_LAYERS.register_module('Deconv', module=nn.ConvTranspose2d)
# TODO: octave conv
