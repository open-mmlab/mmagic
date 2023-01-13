# Copyright (c) OpenMMLab. All rights reserved.
from .basic_conditional_dataset import BasicConditionalDataset
from .basic_frames_dataset import BasicFramesDataset
from .basic_image_dataset import BasicImageDataset
from .cifar10_dataset import CIFAR10
from .comp1k_dataset import AdobeComp1kDataset
from .grow_scale_image_dataset import GrowScaleImgDataset
from .imagenet_dataset import ImageNet
from .mscoco_dataset import MSCoCoDataset
from .paired_image_dataset import PairedImageDataset
from .singan_dataset import SinGANDataset
from .unpaired_image_dataset import UnpairedImageDataset

__all__ = [
    'AdobeComp1kDataset', 'BasicImageDataset', 'BasicFramesDataset',
    'BasicConditionalDataset', 'UnpairedImageDataset', 'PairedImageDataset',
    'ImageNet', 'CIFAR10', 'GrowScaleImgDataset', 'SinGANDataset',
    'MSCoCoDataset'
]
