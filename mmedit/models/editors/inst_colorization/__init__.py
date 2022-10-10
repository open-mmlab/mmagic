# Copyright (c) OpenMMLab. All rights reserved.
from .inst_colorization import InstColorization
from .inst_colorization_net import (FusionGenerator, InstanceGenerator,
                                    SIGGRAPHGenerator)

__all__ = [
    'InstColorization', 'SIGGRAPHGenerator', 'InstanceGenerator',
    'FusionGenerator'
]
