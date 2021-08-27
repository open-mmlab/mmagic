# Copyright (c) OpenMMLab. All rights reserved.
from .augmentation import (BinarizeImage, CopyValueFromKey, Flip,
                           GenerateFrameIndices,
                           GenerateFrameIndiceswithPadding,
                           GenerateSegmentIndices, MirrorSequence, Pad,
                           RandomAffine, RandomJitter, RandomMaskDilation,
                           RandomTransposeHW, Resize, RoundClipZeroOne,
                           TemporalReverse)
from .compose import Compose
from .crop import (Crop, CropAroundCenter, CropAroundFg, CropAroundUnknown,
                   CropLike, FixedCrop, ModCrop, PairedRandomCrop)
from .formating import (Collect, FormatTrimap, GetMaskedImage, ImageToTensor,
                        ToTensor)
from .generate_assistant import GenerateCoordinateAndCell, GenerateHeatmap
from .loading import (GetSpatialDiscountMask, LoadImageFromFile,
                      LoadImageFromFileList, LoadMask, LoadPairedImageFromFile,
                      RandomLoadResizeBg)
from .matting_aug import (CompositeFg, GenerateSeg, GenerateSoftSeg,
                          GenerateTrimap, GenerateTrimapWithDistTransform,
                          MergeFgAndBg, PerturbBg, TransformTrimap)
from .normalization import Normalize, RescaleToZeroOne
from .random_degradations import (AddRandomBlur, AddRandomJPEGCompression,
                                  AddRandomNoise, DegradationsWithShuffle,
                                  RandomResize)
from .random_down_sampling import RandomDownSampling

__all__ = [
    'Collect', 'FormatTrimap', 'LoadImageFromFile', 'LoadMask',
    'RandomLoadResizeBg', 'Compose', 'ImageToTensor', 'ToTensor',
    'GetMaskedImage', 'BinarizeImage', 'Flip', 'Pad', 'RandomAffine',
    'RandomJitter', 'RandomMaskDilation', 'RandomTransposeHW', 'Resize',
    'Crop', 'CropAroundCenter', 'CropAroundUnknown', 'ModCrop',
    'PairedRandomCrop', 'Normalize', 'RescaleToZeroOne', 'GenerateTrimap',
    'MergeFgAndBg', 'CompositeFg', 'TemporalReverse', 'LoadImageFromFileList',
    'GenerateFrameIndices', 'GenerateFrameIndiceswithPadding', 'FixedCrop',
    'LoadPairedImageFromFile', 'GenerateSoftSeg', 'GenerateSeg', 'PerturbBg',
    'CropAroundFg', 'GetSpatialDiscountMask', 'RandomDownSampling',
    'GenerateTrimapWithDistTransform', 'TransformTrimap',
    'GenerateCoordinateAndCell', 'GenerateSegmentIndices', 'MirrorSequence',
    'CropLike', 'GenerateHeatmap', 'CopyValueFromKey', 'RoundClipZeroOne',
    'AddRandomBlur', 'AddRandomJPEGCompression', 'AddRandomNoise',
    'DegradationsWithShuffle', 'RandomResize'
]
