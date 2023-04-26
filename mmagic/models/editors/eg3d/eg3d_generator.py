# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from typing import Optional, Tuple

import torch
from mmengine.model import BaseModule
from torch import Tensor

from mmagic.registry import MODELS
from .eg3d_modules import SuperResolutionModule, TriPlaneBackbone
from .ray_sampler import sample_rays
from .renderer import EG3DRenderer


@MODELS.register_module('EG3DGenerator')
@MODELS.register_module()
class TriplaneGenerator(BaseModule):
    """The generator for EG3D.

    EG3D generator contains three components:

    * A StyleGAN2 based backbone to generate a triplane feature
    * A neural renderer to sample and render low-resolution 2D feature and
      image from generated triplane feature
    * A super resolution module to upsample low-resolution image to
      high-resolution one

    Args:
        out_size (int): The resolution of the generated 2D image.
        noise_size (int): The size of the noise vector of the StyleGAN2
            backbone. Defaults to 512.
        style_channels (int): The number of channels for style code.
            Defaults to 512.
        cond_size (int): The size of the conditional input. Defaults to 25
            (first 16 elements are flattened camera-to-world matrix and the
            last 9 elements are flattened intrinsic matrix).
        cond_mapping_channels (Optional[int]): The channels of the
            conditional mapping layers. If not passed, will use the same value
            as :attr:`style_channels`. Defaults to None.
        cond_scale (float): The scale factor is multiple by the conditional
            input. Defaults to 1.
        zero_cond_input (bool): Whether use 'zero tensor' as the conditional
            input. Defaults to False.
        num_mlps (int): The number of MLP layers (mapping network) used in
            backbone. Defaults to 8.
        triplane_size (int): The size of generated triplane feature. Defaults
            to 256.
        triplane_channels (int): The number of channels for each plane of the
            triplane feature. Defaults to 32.
        sr_in_size (int): The input resolution of super resolution module. If
            the input feature not match with the passed `sr_in_size`, bilinear
            interpolation will be used to resize feature to target size.
            Defaults to 64.
        sr_in_channels (int): The number of the input channels of super
            resolution module. Defaults to 32.
        sr_hidden_channels (int): The number of the hidden channels of super
            resolution module. Defaults to 128.
        sr_out_channels (int): The number of the output channels of super
            resolution module. Defaults to 64.
        sr_add_noise (bool): Whether use noise injection to super resolution
            module. Defaults to False.
        neural_rendering_resolution (int): The resolution of the neural
            rendering output. Defaults to 64. Noted that in the training
            process, neural rendering resolution will be changed.
            Defaults to 64.
        renderer_cfg (int): The config to build :class:`EG3DRenderer`.
            Defaults to '{}'.
        rgb2bgr (bool): Whether convert the RGB output to BGR. This is useful
            when pretrained model is trained on RGB dataset. Defaults to False.
        init_cfg (Optional[dict]): Initialization config. Defaults to None.
    """

    def __init__(self,
                 out_size: int,
                 noise_size: int = 512,
                 style_channels: int = 512,
                 cond_size: int = 25,
                 cond_mapping_channels: Optional[int] = None,
                 cond_scale: float = 1,
                 zero_cond_input: bool = False,
                 num_mlps: int = 8,
                 triplane_size: int = 256,
                 triplane_channels: int = 32,
                 sr_in_size: int = 64,
                 sr_in_channels: int = 32,
                 sr_hidden_channels: int = 128,
                 sr_out_channels: int = 64,
                 sr_antialias: bool = True,
                 sr_add_noise: bool = True,
                 neural_rendering_resolution: int = 64,
                 renderer_cfg: dict = dict(),
                 rgb2bgr: bool = False,
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg=init_cfg)

        self.out_size = out_size
        self.noise_size = noise_size
        self.cond_size = cond_size
        self.style_size = style_channels

        self.cond_scale = cond_scale
        self.zero_cond_input = zero_cond_input
        self.sr_add_noise = sr_add_noise

        # build StyleGAN2 backbone
        self.triplane_channels = triplane_channels
        self.backbone = TriPlaneBackbone(
            out_size=triplane_size,
            noise_size=noise_size,
            out_channels=triplane_channels * 3,
            style_channels=style_channels,
            num_mlps=num_mlps,
            cond_size=cond_size,
            cond_scale=self.cond_scale,
            cond_mapping_channels=cond_mapping_channels,
            zero_cond_input=self.zero_cond_input)

        # build renderer + nerf-decoder and ray sampler
        self.neural_rendering_resolution = neural_rendering_resolution
        renderer_cfg_ = deepcopy(renderer_cfg)
        decoder_cfg_ = renderer_cfg.get('decoder_cfg', dict())
        decoder_cfg_['in_channels'] = triplane_channels
        decoder_cfg_['out_channels'] = sr_in_channels
        renderer_cfg_['decoder_cfg'] = decoder_cfg_
        self.renderer = EG3DRenderer(**deepcopy(renderer_cfg_))

        # build super-resolution module
        sr_factor = out_size // sr_in_size
        assert sr_factor in [
            2, 4, 8
        ], ('Only support super resolution with factor 2, 4 or 8. '
            f'But \'out_size\' and \'sr_in_size\'are {out_size} and '
            f'{sr_in_size}.')
        self.sr_model = SuperResolutionModule(
            in_channels=sr_in_channels,
            in_size=sr_in_size,
            hidden_size=sr_in_size * 2 if sr_factor in [4, 8] else sr_in_size,
            out_size=out_size,
            style_channels=style_channels,
            hidden_channels=sr_hidden_channels,
            out_channels=sr_out_channels,
            sr_antialias=sr_antialias)

        # flag for pretrained models
        self.rgb2bgr = rgb2bgr

    def sample_ray(self, cond: torch.Tensor) -> Tuple[Tensor]:
        """Sample render points corresponding to the given conditional.

        Args:
            cond (torch.Tensor): Conditional inputs.

        Returns:
            Tuple[Tensor]: The original and direction vector of sampled rays.
        """
        cam2world_matrix = cond[:, :16].view(-1, 4, 4)
        intrinsics = cond[:, 16:25].view(-1, 3, 3)

        ray_origin, ray_directions = sample_rays(
            cam2world_matrix, intrinsics, self.neural_rendering_resolution)
        return ray_origin, ray_directions

    def forward(self,
                noise: Tensor,
                label: Optional[Tensor] = None,
                truncation: Optional[float] = 1,
                num_truncation_layer: Optional[int] = None,
                input_is_latent: bool = False,
                plane: Optional[Tensor] = None,
                add_noise: bool = True,
                randomize_noise: bool = True,
                render_kwargs: Optional[dict] = None) -> dict:
        """The forward function for EG3D generator.

        Args:
            noise (Tensor): The input noise vector.
            label (Optional[Tensor]): The conditional input. Defaults to None.
            truncation (float, optional): Truncation factor. Give value less
                than 1., the truncation trick will be adopted. Defaults to 1.
            num_truncation_layer (int, optional): Number of layers use
                truncated latent. Defaults to None.
            input_is_latent (bool): Whether the input latent. Defaults to
                False.
            plane (Optional[Tensor]): The pre-generated triplane feature. If
                passed, will use the passed plane to generate 2D image.
                Defaults to None.
            add_noise (bool): Whether apply noise injection to the triplane
                backbone. Defaults to True.
            randomize_noise (bool, optional): If `False`, images are sampled
                with the buffered noise tensor injected to the style conv
                block. Defaults to True.
            render_kwargs (Optional[dict], optional): The specific kwargs for
                rendering. Defaults to None.

        Returns:
            dict: A dict contains 'fake_img', 'lr_img', 'depth',
                'ray_directions' and 'ray_origins'.
        """
        batch_size = noise.shape[0]

        if not input_is_latent:
            styles = self.backbone.mapping(
                noise,
                label,
                truncation=truncation,
                num_truncation_layer=num_truncation_layer)
        else:
            styles = noise

        ray_origins, ray_directions = self.sample_ray(label)

        if plane is None:
            plane = self.backbone.synthesis(
                styles, add_noise=add_noise, randomize_noise=randomize_noise)

        # Reshape output into three `triplane_channels`-channel planes
        plane = plane.view(
            len(plane), 3, self.triplane_channels, plane.shape[-2],
            plane.shape[-1])

        # Perform volume rendering
        feature_samples, depth_samples, _ = self.renderer(
            plane, ray_origins, ray_directions, render_kwargs=render_kwargs)

        # Reshape into 'raw' neural-rendered image
        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(
            batch_size, feature_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2,
                                            1).reshape(batch_size, 1, H, W)

        # Run super resolution to get final image
        rgb_image = feature_image[:, :3]
        sr_image = self.sr_model(
            rgb_image, feature_image, styles, add_noise=self.sr_add_noise)

        if self.rgb2bgr:
            sr_image = sr_image.flip(1)
            rgb_image = rgb_image.flip(1)

        output_dict = dict(
            fake_img=sr_image,
            lr_img=rgb_image,  # low-resolution images
            depth=depth_image,
            ray_directions=ray_directions,
            ray_origins=ray_origins)

        return output_dict
