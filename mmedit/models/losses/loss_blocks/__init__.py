# Copyright (c) OpenMMLab. All rights reserved.
from .clip_loss_block import CLIPLossBlock
from .disc_auxiliary_loss_blocks import (DiscShiftLossBlock,
                                         GradientPenaltyLossBlock,
                                         R1GradientPenaltyBlock)
from .face_id_loss_block import FaceIdLossBlock
from .gan_loss_blocks import GANLossBlocks
from .gen_auxiliary_loss_blocks import GeneratorPathRegularizerBlock

__all__ = [
    'DiscShiftLossBlock', 'GradientPenaltyLossBlock', 'R1GradientPenaltyBlock',
    'GANLossBlocks', 'GeneratorPathRegularizerBlock', 'CLIPLossBlock',
    'FaceIdLossBlock'
]
