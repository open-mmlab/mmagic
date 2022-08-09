# Copyright (c) OpenMMLab. All rights reserved.
from .alpha import GenerateSeg, GenerateSoftSeg
from .aug_frames import MirrorSequence, TemporalReverse
from .aug_pixel import (BinarizeImage, Clip, ColorJitter, RandomAffine,
                        RandomMaskDilation, UnsharpMasking)
from .aug_shape import Flip, RandomRotation, RandomTransposeHW, Resize
from .crop import (Crop, CropAroundCenter, CropAroundFg, CropAroundUnknown,
                   CropLike, FixedCrop, ModCrop, PairedRandomCrop,
                   RandomResizedCrop)
from .fgbg import (CompositeFg, MergeFgAndBg, PerturbBg, RandomJitter,
                   RandomLoadResizeBg)
from .formatting import PackEditInputs, ToTensor
from .generate_assistant import (GenerateCoordinateAndCell,
                                 GenerateFacialHeatmap)
from .generate_frame_indices import (GenerateFrameIndices,
                                     GenerateFrameIndiceswithPadding,
                                     GenerateSegmentIndices)
from .get_masked_image import GetMaskedImage
from .loading import GetSpatialDiscountMask, LoadImageFromFile, LoadMask
from .matlab_like_resize import MATLABLikeResize
from .normalization import Normalize, RescaleToZeroOne
from .random_degradations import (DegradationsWithShuffle, RandomBlur,
                                  RandomJPEGCompression, RandomNoise,
                                  RandomResize, RandomVideoCompression)
from .random_down_sampling import RandomDownSampling
from .trans_utils import (adjust_gamma, bbox2mask, brush_stroke_mask,
                          get_irregular_mask, random_bbox)
from .trimap import (FormatTrimap, GenerateTrimap,
                     GenerateTrimapWithDistTransform, TransformTrimap)
from .values import CopyValues, SetValues

__all__ = [
    'random_bbox', 'get_irregular_mask', 'brush_stroke_mask', 'bbox2mask',
    'adjust_gamma', 'BinarizeImage', 'Clip', 'ColorJitter', 'CopyValues',
    'Crop', 'CropLike', 'DegradationsWithShuffle', 'LoadImageFromFile',
    'LoadMask', 'Flip', 'FixedCrop', 'GenerateCoordinateAndCell',
    'GenerateFacialHeatmap', 'GenerateFrameIndices',
    'GenerateFrameIndiceswithPadding', 'GenerateSegmentIndices',
    'GetMaskedImage', 'GetSpatialDiscountMask', 'MATLABLikeResize',
    'MirrorSequence', 'ModCrop', 'Normalize', 'PackEditInputs',
    'PairedRandomCrop', 'RandomAffine', 'RandomBlur', 'RandomDownSampling',
    'RandomJPEGCompression', 'RandomMaskDilation', 'RandomNoise',
    'RandomResize', 'RandomResizedCrop', 'RandomRotation', 'RandomTransposeHW',
    'RandomVideoCompression', 'RescaleToZeroOne', 'Resize', 'SetValues',
    'TemporalReverse', 'ToTensor', 'UnsharpMasking', 'CropAroundCenter',
    'CropAroundFg', 'GenerateSeg', 'CropAroundUnknown', 'GenerateSoftSeg',
    'FormatTrimap', 'TransformTrimap', 'GenerateTrimap',
    'GenerateTrimapWithDistTransform', 'CompositeFg', 'RandomLoadResizeBg',
    'MergeFgAndBg', 'PerturbBg', 'RandomJitter'
]
