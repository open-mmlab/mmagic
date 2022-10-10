# Copyright (c) OpenMMLab. All rights reserved.
from .inst_colorization import INSTA
from .inst_colorizatiuion_net import (FusionGenerator, InstanceGenerator,
                                      SIGGRAPHGenerator)

__all__ = [
    'INSTA', 'SIGGRAPHGenerator', 'InstanceGenerator', 'FusionGenerator'
]
