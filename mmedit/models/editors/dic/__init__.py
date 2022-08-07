# Copyright (c) OpenMMLab. All rights reserved.
from .dic import DIC
from .dic_net import DICNet
from .feedback_hour_glass import FeedbackHourglass
from .light_cnn import LightCNN

__all__ = [
    'DICNet',
    'DIC',
    'FeedbackHourglass',
    'LightCNN',
]
