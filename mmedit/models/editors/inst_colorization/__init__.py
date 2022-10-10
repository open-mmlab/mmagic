# Copyright (c) OpenMMLab. All rights reserved.
from .insta import INSTA
from .insta_net import FusionGenerator, InstanceGenerator, SIGGRAPHGenerator

__all__ = [
    'INSTA', 'SIGGRAPHGenerator', 'InstanceGenerator', 'FusionGenerator'
]
