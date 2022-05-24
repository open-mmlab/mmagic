# Copyright (c) OpenMMLab. All rights reserved.
from .aug_frames import MirrorSequence, TemporalReverse
from .aug_matting import (CompositeFg, GenerateSeg, GenerateSoftSeg,
                          GenerateTrimap, GenerateTrimapWithDistTransform,
                          MergeFgAndBg, PerturbBg, TransformTrimap)
from .aug_pixel import (BinarizeImage, Clip, ColorJitter, RandomAffine,
                        RandomJitter, RandomMaskDilation, UnsharpMasking)
from .aug_shape import Flip, RandomRotation, RandomTransposeHW, Resize
from .crop import (Crop, CropAroundCenter, CropAroundFg, CropAroundUnknown,
                   CropLike, FixedCrop, ModCrop, PairedRandomCrop,
                   RandomResizedCrop)
from .formatting import PackEditInputs, ToTensor
from .generate_frame_indices import (GenerateFrameIndices,
                                     GenerateFrameIndiceswithPadding,
                                     GenerateSegmentIndices)
from .loading import LoadImageFromFile
from .matlab_like_resize import MATLABLikeResize
from .random_degradations import (DegradationsWithShuffle, RandomBlur,
                                  RandomJPEGCompression, RandomNoise,
                                  RandomResize, RandomVideoCompression)
from .random_down_sampling import RandomDownSampling
from .values import CopyValues

__all__ = [
    'BinarizeImage',
    'Clip',
    'ColorJitter',
    'CopyValues',
    'Crop',
    'CropAroundCenter',
    'CropAroundFg',
    'CropAroundUnknown',
    'CropLike',
    'DegradationsWithShuffle',
    'LoadImageFromFile',
    'Flip',
    'FixedCrop',
    'GenerateFrameIndices',
    'GenerateFrameIndiceswithPadding',
    'GenerateSegmentIndices',
    'MATLABLikeResize',
    'MirrorSequence',
    'ModCrop',
    'PackEditInputs',
    'PairedRandomCrop',
    'RandomAffine',
    'RandomBlur',
    'RandomDownSampling',
    'RandomJitter',
    'RandomJPEGCompression',
    'RandomMaskDilation',
    'RandomNoise',
    'RandomResize',
    'RandomResizedCrop',
    'RandomRotation',
    'RandomTransposeHW',
    'RandomVideoCompression',
    'Resize',
    'TemporalReverse',
    'ToTensor',
    'UnsharpMasking',
    # matting
    'CompositeFg',
    'GenerateSeg',
    'GenerateSoftSeg',
    'GenerateTrimap',
    'GenerateTrimapWithDistTransform',
    'MergeFgAndBg',
    'PerturbBg',
    'TransformTrimap'
]
