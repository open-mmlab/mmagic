# Copyright (c) OpenMMLab. All rights reserved.
from .modified_vgg import ModifiedVGG
from .sr_resnet import MSRResNet
from .srgan import SRGAN

__all__ = [
    'ModifiedVGG',
    'MSRResNet',
    'SRGAN',
]
