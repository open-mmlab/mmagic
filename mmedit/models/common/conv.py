from mmcv.cnn import CONV_LAYERS
from torch import nn as nn

CONV_LAYERS.register_module('Deconv', module=nn.ConvTranspose2d)
# TODO: octave conv
