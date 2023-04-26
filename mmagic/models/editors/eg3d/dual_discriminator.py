# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.runner.amp import autocast
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION
from torch import Tensor

from mmagic.registry import MODELS
from ..stylegan2 import StyleGAN2Discriminator


@MODELS.register_module('EG3DDiscriminator')
@MODELS.register_module()
class DualDiscriminator(StyleGAN2Discriminator):
    """Dual Discriminator for EG3D. DualDiscriminator shares the same network
    structure with StyleGAN2's Discriminator. However, DualDiscriminator take
    volume rendered low-resolution image and super-resolved image at the same
    time. The LR image will be upsampled and concatenate with SR ones, and then
    feed to the discriminator together.

    Args:
        img_channels (int): The number of the image channels. Defaults to 3.
        use_dual_disc (bool): Whether use dual discriminator as EG3D. If True,
            the input channel of the first conv block will be set as
            `2 * img_channels`. Defaults to True.
        disc_c_noise (float): The factor of noise's standard deviation add to
            conditional input before passed to mapping network. Defaults to 0.
        *args, **kwargs: Arguments for StyleGAN2Discriminator.
    """

    def __init__(self,
                 img_channels: int = 3,
                 use_dual_disc: bool = True,
                 disc_c_noise: float = 0,
                 *args,
                 **kwargs):
        if use_dual_disc:
            img_channels *= 2
        self.use_dual_disc = use_dual_disc
        super().__init__(img_channels=img_channels, *args, **kwargs)
        self.disc_c_noise = disc_c_noise

    def forward(self,
                img: Tensor,
                img_raw: Optional[Tensor] = None,
                cond: Optional[Tensor] = None):
        """Forward function.

        Args:
            img (torch.Tensor): Input high resoluation image tensor.
            img_raw (torch.Tensor): Input raw (low resolution) image tensor.
                Defaults to None.
            cond (torch.Tensor): The conditional input (camera-to-world matrix
                and intrinsics matrix). Defaults to None.

        Returns:
            torch.Tensor: Predict score for the input image.
        """
        if self.use_dual_disc:
            assert img_raw is not None, (
                '\'img_raw\' must be passed when \'use_dual_disc\' is True.')

        # This setting was used to finetune on converted weights
        if self.input_bgr2rgb:
            img = img[:, [2, 1, 0], ...]
            if img_raw is not None:
                img_raw = img_raw[:, [2, 1, 0], ...]

        if img_raw is not None:
            # the official implementation only use 'antialiased' upsampline,
            # therefore we only support 'antialiased' for torch >= 1.11.0
            interpolation_kwargs = dict(
                size=(img.shape[-1], img.shape[-1]),
                mode='bilinear',
                align_corners=False)
            if digit_version(TORCH_VERSION) >= digit_version('1.11.0'):
                interpolation_kwargs['antialias'] = True
            img_raw_sr = F.interpolate(img_raw, **interpolation_kwargs)
            img = torch.cat([img, img_raw_sr], dim=1)

        # convs has own fp-16 controller, do not wrap here
        x = self.convs(img)
        x = self.mbstd_layer(x)

        fp16_enabled = (
            self.final_conv.fp16_enabled or not self.convert_input_fp32)
        with autocast(enabled=fp16_enabled):
            if not fp16_enabled:
                x = x.to(torch.float32)
            x = self.final_conv(x)
            x = x.view(x.shape[0], -1)
            x = self.final_linear(x)

            # conditioning
            if cond is not None:
                assert self.mapping is not None, (
                    '\'mapping\' network must not be None when conditional '
                    'input is passed.')

                # if self.disc_c_noise is not None and self.disc_c_noise > 0:
                if self.disc_c_noise is not None:
                    cond = cond + torch.randn_like(
                        cond) * cond.std() * self.disc_c_noise
                cmap = self.mapping(None, cond)
                x = (x * cmap).sum(
                    dim=1, keepdim=True) * (1 / np.sqrt(cmap.shape[1]))
        return x
