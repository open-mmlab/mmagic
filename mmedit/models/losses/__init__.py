# Copyright (c) OpenMMLab. All rights reserved.
from .clip_loss import CLIPLoss
from .composition_loss import (CharbonnierCompLoss, L1CompositionLoss,
                               MSECompositionLoss)
from .face_id_loss import FaceIdLoss
from .feature_loss import LightCNNFeatureLoss
from .gan_loss import (DiscShiftLoss, GANLoss, GaussianBlur,
                       GradientPenaltyLoss, disc_shift_loss,
                       gen_path_regularizer, gradient_penalty_loss,
                       r1_gradient_penalty_loss)
from .gradient_loss import GradientLoss
from .loss_comps import (CLIPLossComps, DiscShiftLossComps, FaceIdLossComps,
                         GANLossComps, GeneratorPathRegularizerComps,
                         GradientPenaltyLossComps, R1GradientPenaltyComps)
from .loss_wrapper import mask_reduce_loss, reduce_loss
from .perceptual_loss import (PerceptualLoss, PerceptualVGG,
                              TransferalPerceptualLoss)
from .pixelwise_loss import (CharbonnierLoss, L1Loss, MaskedTVLoss, MSELoss,
                             PSNRLoss, tv_loss)

__all__ = [
    'L1Loss', 'MSELoss', 'CharbonnierLoss', 'L1CompositionLoss',
    'MSECompositionLoss', 'CharbonnierCompLoss', 'GANLoss', 'GaussianBlur',
    'GradientPenaltyLoss', 'PerceptualLoss', 'PerceptualVGG', 'reduce_loss',
    'mask_reduce_loss', 'DiscShiftLoss', 'MaskedTVLoss', 'GradientLoss',
    'TransferalPerceptualLoss', 'LightCNNFeatureLoss', 'gradient_penalty_loss',
    'r1_gradient_penalty_loss', 'gen_path_regularizer', 'FaceIdLoss',
    'CLIPLoss', 'CLIPLossComps', 'DiscShiftLossComps', 'FaceIdLossComps',
    'GANLossComps', 'GeneratorPathRegularizerComps',
    'GradientPenaltyLossComps', 'R1GradientPenaltyComps', 'disc_shift_loss',
    'tv_loss', 'PSNRLoss'
]
