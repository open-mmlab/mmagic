# Copyright (c) OpenMMLab. All rights reserved.
from .basic_frames_dataset import BasicFramesDataset
from .basic_image_dataset import BasicImageDataset
from .comp1k_dataset import AdobeComp1kDataset

__all__ = [
    'AdobeComp1kDataset',
    'BasicImageDataset',
    'BasicFramesDataset',
]
