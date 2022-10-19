# Copyright (c) OpenMMLab. All rights reserved.
from .colorization_net import (FusionGenerator, InstanceGenerator,
                               SIGGRAPHGenerator)
from .inst_colorization import InstColorization
from .inst_colorization_net import InstColorizationGenerator

__all__ = [
    'InstColorization', 'SIGGRAPHGenerator', 'InstanceGenerator',
    'FusionGenerator', 'InstColorizationGenerator'
]
