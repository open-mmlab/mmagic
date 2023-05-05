# Copyright (c) OpenMMLab. All rights reserved.
from .colorization_net import ColorizationNet
from .fusion_net import FusionNet
from .inst_colorization import InstColorization

__all__ = [
    'InstColorization',
    'ColorizationNet',
    'FusionNet',
]
