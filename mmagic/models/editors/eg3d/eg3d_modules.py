# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from mmengine import print_log
from mmengine.model import BaseModule
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION

from ..stylegan2 import StyleGAN2Generator
from ..stylegan2.stylegan2_modules import ModulatedStyleConv, ModulatedToRGB


class TriPlaneBackbone(StyleGAN2Generator):
    """Tr-plane backbone for EG3D generator. This class is a wrapper of
    StyleGAN2Generator.

    Args:
        noise_size (int, optional): The size of (number of channels) the input
            noise. If not passed, will be set the same value as
            :attr:`style_channels`. Defaults to None.
        out_size (int): The output size of the StyleGAN2 generator.
        out_channels (int): The number of channels for output.
        num_mlps (int, optional): The number of MLP layers. Defaults to 8.
        style_channels (int): The number of channels for style code. Defaults
            to 512.
        cond_size (int, optional): The size of the conditional input. If not
            passed or less than 1, no conditional embedding will be used.
            Defaults to None.
        cond_mapping_channels (int, optional): The channels of the
            conditional mapping layers. If not passed, will use the same value
            as :attr:`style_channels`. Defaults to None.
        cond_scale (float): The scale factor is multiple by the conditional
            input. Defaults to 1.
        zero_cond_input (bool): Whether use 'zero tensor' as the conditional
            input. Defaults to False.

        *args, **kwargs: Arguments for StyleGAN2Generator.
    """

    def __init__(self,
                 noise_size: int,
                 out_size: int,
                 out_channels: int,
                 num_mlps: int = 8,
                 style_channels: int = 512,
                 cond_size: Optional[int] = 25,
                 cond_mapping_channels: Optional[int] = None,
                 cond_scale: float = 0,
                 zero_cond_input: bool = False,
                 *args,
                 **kwargs):
        super().__init__(
            out_size,
            style_channels,
            num_mlps=num_mlps,
            noise_size=noise_size,
            out_channels=out_channels,
            cond_size=cond_size,
            cond_mapping_channels=cond_mapping_channels,
            *args,
            **kwargs)

        self.cond_scale = cond_scale
        self.zero_cond_input = zero_cond_input

    def mapping(self,
                noise: torch.Tensor,
                label: Optional[torch.Tensor] = None,
                truncation: float = 1,
                num_truncation_layer: Optional[int] = None,
                update_ws: bool = True) -> torch.Tensor:
        """Mapping input noise (z) to style space (w).

        Args:
            noise (torch.Tensor): Noise input.
            label (Optional[torch.Tensor]): Conditional inputs.
                Defaults to None.
            truncation (float, optional): Truncation factor. Give value less
                than 1., the truncation trick will be adopted. Defaults to 1.
            num_truncation_layer (int, optional): Number of layers use
                truncated latent. Defaults to None.
            update_ws (bool): Whether update latent code with EMA. Only work
                when `w_avg` is registered. Defaults to False.

        Returns:
            torch.Tensor: Style codes after mapping.
        """
        assert noise.shape[1] == self.noise_size
        noise = self.pixel_norm(noise)

        if label is not None:
            assert label.shape[1] == self.cond_size
            if self.zero_cond_input:
                label = torch.zeros_like(label)
            label = label * self.cond_scale
            embedding = self.embed(label)
            embedding = self.pixel_norm(embedding)
        else:
            # generate a zero input even if cond is not passed.
            if self.zero_cond_input:
                assert self.cond_size is not None, (
                    '\'cond_size\' must be passed when '
                    '\'zero_cond_input\' is True.')
                label = torch.zeros(
                    noise.shape[0], self.cond_size, device=noise.device)
                embedding = self.embed(label)
                embedding = self.pixel_norm(embedding)
            else:
                embedding = None
        mapping_input = noise if embedding is None \
            else torch.cat([noise, embedding], dim=1)

        styles = self.style_mapping(mapping_input)

        if hasattr(self, 'w_avg') and update_ws:
            self.w_avg.copy_(styles.detach().mean(
                dim=0).lerp(self.w_avg, self.w_avg_beta))

        if truncation < 1:
            truncation_latent = self.get_mean_latent()
            styles = truncation_latent + truncation * (
                styles - truncation_latent)

        return styles

    def synthesis(self, styles: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Generate the Triplane feature.

        Args:
            styles (torch.Tensor): The input style code.
            *args, **kwargs: Arguments for StyleGAN2Generator's forward.

        Returns:
            torch.Tensor: The generated Triplane feature.
        """
        outputs = super().forward(
            styles, input_is_latent=True, update_ws=False, *args, **kwargs)
        return outputs


class SuperResolutionModule(BaseModule):
    """Super resolution module for EG3D generator.

    Args:
        in_channels (int): The channels of the input feature.
        in_size (int): The size of the input feature.
        hidden_size (int): The size of the hidden feature. Only support hidden
            size equals to in_size or in_size times two. Defaults to None.
        out_size (int): The size of the output image. Defaults to None.
        hidden_channels (int): The channels of the hidden feature.
            Defaults to 64.
        style_channels (int): The channels of the style code. Defaults to 512.
        sr_antialias (bool): Whether use antialias interpolation method in
            upsampling. Defaults to True.
        fp16_enable (bool): Whether enable fp16 in this module.
            Defaults to False.
    """

    def __init__(self,
                 in_channels: int,
                 in_size: Optional[int] = None,
                 hidden_size: Optional[int] = None,
                 out_size: Optional[int] = None,
                 hidden_channels: int = 128,
                 out_channels: int = 64,
                 style_channels: Optional[int] = 512,
                 sr_antialias: bool = True,
                 fp16_enable: bool = False):
        super().__init__()
        self.in_size = in_size
        self.sr_antialias = sr_antialias
        self.fp16_enable = fp16_enable

        self.style_channels = style_channels

        block0_upsample = hidden_size > in_size
        if block0_upsample:
            assert hidden_size == in_size * 2, (
                'Only support upsampling with factor 2. But \'in_resolution\' '
                f'and \'hidden_resolution\' are \'{in_size}\' and '
                f'\'{hidden_size}\'.')

        self.block0 = SynthesisBlock(
            in_channels,
            hidden_channels,
            style_channels,
            img_channels=3,
            upsample=block0_upsample,
            fp16_enabled=fp16_enable)

        block1_upsample = out_size > hidden_size
        if block1_upsample:
            assert out_size == hidden_size * 2, (
                'Only support upsampling with factor 2. But '
                '\'hidden_resolution\' and \'out_resolution\' are '
                f'\'{in_size}\' and \'{hidden_size}\'.')
        self.block1 = SynthesisBlock(
            hidden_channels,
            out_channels,
            style_channels,
            img_channels=3,
            upsample=block1_upsample,
            fp16_enabled=fp16_enable)
        if digit_version(TORCH_VERSION) < digit_version('1.11.0'):
            print_log(f'Current Pytorch version is {TORCH_VERSION}, lower '
                      'than 1.11.0. \'sr_antialias\' is ignored.')

    def forward(self,
                img: torch.Tensor,
                feature: torch.Tensor,
                styles: Union[torch.Tensor, List[torch.Tensor]],
                add_noise: bool = False) -> torch.Tensor:
        """Forward function.

        Args:
            img (torch.Tensor): Image to super resolution.
            x (torch.Tensor): Feature map of the input image.
            styles (torch.Tensor): Style codes in w space.
            add_noise (bool, optional): Whether add noise to image.
                Defaults to False.

        Returns:
            torch.Tensor: Image after super resolution.
        """
        if isinstance(styles, list):
            styles = styles[-1]
        if styles.ndim == 3:
            styles = styles[-1]
        assert styles.ndim == 2 and styles.shape[-1] == self.style_channels
        styles = styles[:, None, :].repeat(1, 3, 1)
        if feature.shape[-1] != self.in_size:
            interpolation_kwargs = dict(
                size=(self.in_size, self.in_size),
                mode='bilinear',
                align_corners=False)
            if digit_version(TORCH_VERSION) >= digit_version('1.11.0'):
                interpolation_kwargs['antialias'] = self.sr_antialias
            feature = F.interpolate(feature, **interpolation_kwargs)
            img = F.interpolate(img, **interpolation_kwargs)
        feature, img = self.block0(feature, img, styles, add_noise)
        feature, img = self.block1(feature, img, styles, add_noise)
        return img


class SynthesisBlock(BaseModule):
    """Synthesis block for EG3D's SuperResolutionModule.

    Args:
        in_channels (int): The number of channels for the input feature.
        out_channels (int): The number of channels for the output feature.
        style_channels (int): The number of channels for style code.
        img_channels (int): The number of channels of output image.
        upsample (bool): Whether do upsampling. Defaults to True.
        conv_clamp (float, optional): Whether clamp the convolutional layer
            results to avoid gradient overflow. Defaults to `256.0`.
        fp16_enabled (bool): Whether enable fp16. Defaults to False.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 style_channels: int,
                 img_channels: int,
                 upsample: bool = True,
                 conv_clamp: int = 256,
                 fp16_enabled: bool = False):
        super().__init__()
        # architecture is default as 'skip' in EG3D
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fp16_enabled = fp16_enabled

        self.upsample = upsample

        self.conv0 = ModulatedStyleConv(
            in_channels,
            out_channels,
            kernel_size=3,
            upsample=upsample,
            style_channels=style_channels,
            fp16_enabled=fp16_enabled,
            conv_clamp=conv_clamp)

        self.conv1 = ModulatedStyleConv(
            out_channels,
            out_channels,
            kernel_size=3,
            style_channels=style_channels,
            conv_clamp=conv_clamp)

        self.to_rgb = ModulatedToRGB(
            out_channels, style_channels, img_channels, upsample=upsample)

    def forward(self,
                x: torch.Tensor,
                img: torch.Tensor,
                styles: torch.Tensor,
                add_noise: bool = False) -> Tuple[torch.Tensor]:
        """Forward Synthesis block.

        Args:
            x (torch.Tensor): Input feature.
            img (torch.Tensor): Input image.
            styles (torch.Tensor): Input style code.
            add_noise (bool, optional): Whether apply noise injection.
                Defaults to False.

        Returns:
            Tuple[torch.Tensor]: Output feature and image.
        """
        w_iter = iter(styles.unbind(dim=1))
        x = self.conv0(x, next(w_iter), add_noise=add_noise)
        x = self.conv1(x, next(w_iter), add_noise=add_noise)

        img = self.to_rgb(x, next(w_iter), img)

        assert img.dtype == torch.float32
        return x, img
