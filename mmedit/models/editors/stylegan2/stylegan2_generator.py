# Copyright (c) OpenMMLab. All rights reserved.
import random

import mmengine
import numpy as np
import torch
import torch.nn as nn
from mmengine.runner.amp import autocast
from mmengine.runner.checkpoint import _load_checkpoint_with_prefix

from mmedit.registry import MODULES
from ...utils import get_module_device
from ..pggan import PixelNorm
from ..stylegan1 import (ConstantInput, EqualLinearActModule, get_mean_latent,
                         style_mixing)
from .stylegan2_modules import ModulatedStyleConv, ModulatedToRGB


@MODULES.register_module('StyleGANv2Generator')
@MODULES.register_module()
class StyleGAN2Generator(nn.Module):
    r"""StyleGAN2 Generator.

    In StyleGAN2, we use a static architecture composing of a style mapping
    module and number of convolutional style blocks. More details can be found
    in: Analyzing and Improving the Image Quality of StyleGAN CVPR2020.

    You can load pretrained model through passing information into
    ``pretrained`` argument. We have already offered official weights as
    follows:

    - stylegan2-ffhq-config-f: https://download.openmmlab.com/mmediting/stylegan2/official_weights/stylegan2-ffhq-config-f-official_20210327_171224-bce9310c.pth  # noqa
    - stylegan2-horse-config-f: https://download.openmmlab.com/mmediting/stylegan2/official_weights/stylegan2-horse-config-f-official_20210327_173203-ef3e69ca.pth  # noqa
    - stylegan2-car-config-f: https://download.openmmlab.com/mmediting/stylegan2/official_weights/stylegan2-car-config-f-official_20210327_172340-8cfe053c.pth  # noqa
    - stylegan2-cat-config-f: https://download.openmmlab.com/mmediting/stylegan2/official_weights/stylegan2-cat-config-f-official_20210327_172444-15bc485b.pth  # noqa
    - stylegan2-church-config-f: https://download.openmmlab.com/mmediting/stylegan2/official_weights/stylegan2-church-config-f-official_20210327_172657-1d42b7d1.pth  # noqa

    If you want to load the ema model, you can just use following codes:

    .. code-block:: python

        # ckpt_http is one of the valid path from http source
        generator = StyleGANv2Generator(1024, 512,
                                        pretrained=dict(
                                            ckpt_path=ckpt_http,
                                            prefix='generator_ema'))

    Of course, you can also download the checkpoint in advance and set
    ``ckpt_path`` with local path. If you just want to load the original
    generator (not the ema model), please set the prefix with 'generator'.

    Note that our implementation allows to generate BGR image, while the
    original StyleGAN2 outputs RGB images by default. Thus, we provide
    ``bgr2rgb`` argument to convert the image space.

    Args:
        out_size (int): The output size of the StyleGAN2 generator.
        style_channels (int): The number of channels for style code.
        out_channels (int): The number of channels for output. Defaults to 3.
        noise_size (int, optional): The size of (number of channels) the input
            noise. If not passed, will be set the same value as
            :attr:`style_channels`. Defaults to None.
        cond_size (int, optional): The size of the conditional input. If not
            passed or less than 1, no conditional embedding will be used.
            Defaults to None.
        cond_mapping_channels (int, optional): The channels of the
            conditional mapping layers. If not passed, will use the same value
            as :attr:`style_channels`. Defaults to None.
        num_mlps (int, optional): The number of MLP layers. Defaults to 8.
        channel_multiplier (int, optional): The multiplier factor for the
            channel number. Defaults to 2.
        blur_kernel (list, optional): The blurry kernel. Defaults
            to [1, 3, 3, 1].
        lr_mlp (float, optional): The learning rate for the style mapping
            layer. Defaults to 0.01.
        default_style_mode (str, optional): The default mode of style mixing.
            In training, we defaultly adopt mixing style mode. However, in the
            evaluation, we use 'single' style mode. `['mix', 'single']` are
            currently supported. Defaults to 'mix'.
        eval_style_mode (str, optional): The evaluation mode of style mixing.
            Defaults to 'single'.
        mix_prob (float, optional): Mixing probability. The value should be
            in range of [0, 1]. Defaults to ``0.9``.
        update_mean_latent_with_ema (bool, optional): Whether update mean
            latent code (w) with EMA. Defaults to False.
        w_avg_beta (float, optional): The value used for update `w_avg`.
            Defaults to 0.998.
        num_fp16_scales (int, optional): The number of resolutions to use auto
            fp16 training. Different from ``fp16_enabled``, this argument
            allows users to adopt FP16 training only in several blocks.
            This behaviour is much more similar to the official implementation
            by Tero. Defaults to 0.
        fp16_enabled (bool, optional): Whether to use fp16 training in this
            module. If this flag is `True`, the whole module will be wrapped
            with ``auto_fp16``. Defaults to False.
        pretrained (dict | None, optional): Information for pretained models.
            The necessary key is 'ckpt_path'. Besides, you can also provide
            'prefix' to load the generator part from the whole state dict.
            Defaults to None.
    """

    def __init__(self,
                 out_size,
                 style_channels,
                 out_channels=3,
                 noise_size=None,
                 cond_size=None,
                 cond_mapping_channels=None,
                 num_mlps=8,
                 channel_multiplier=2,
                 blur_kernel=[1, 3, 3, 1],
                 lr_mlp=0.01,
                 default_style_mode='mix',
                 eval_style_mode='single',
                 norm_eps=1e-6,
                 mix_prob=0.9,
                 update_mean_latent_with_ema=False,
                 w_avg_beta=0.998,
                 num_fp16_scales=0,
                 fp16_enabled=False,
                 bgr2rgb=False,
                 pretrained=None):
        super().__init__()
        self.out_size = out_size
        self.style_channels = style_channels
        self.out_channels = out_channels
        self.num_mlps = num_mlps
        self.channel_multiplier = channel_multiplier
        self.lr_mlp = lr_mlp
        self._default_style_mode = default_style_mode
        self.default_style_mode = default_style_mode
        self.eval_style_mode = eval_style_mode
        self.mix_prob = mix_prob
        self.num_fp16_scales = num_fp16_scales
        self.fp16_enabled = fp16_enabled
        self.bgr2rgb = bgr2rgb

        self.noise_size = style_channels if noise_size is None else noise_size

        self.cond_size = cond_size
        if self.cond_size is not None and self.cond_size > 0:
            cond_mapping_channels = style_channels \
                if cond_mapping_channels is None else cond_mapping_channels
            self.embed = EqualLinearActModule(cond_size, cond_mapping_channels)
            # NOTE: conditional input is passed, do 2nd moment norm for
            # embedding and noise input respectively, therefore mapping layer
            # start with FC layer
            mapping_layers = []
        else:
            cond_mapping_channels = 0
            # NOTE: conditional input is not passed, put 2nd moment norm at
            # the start of mapping layers
            mapping_layers = [PixelNorm(eps=norm_eps)]
        in_feat = cond_mapping_channels + self.noise_size

        # define pixel norm
        self.pixel_norm = PixelNorm(eps=norm_eps)

        # define style mapping layers
        for idx in range(num_mlps):
            mapping_layers.append(
                EqualLinearActModule(
                    in_feat if idx == 0 else style_channels,
                    style_channels,
                    equalized_lr_cfg=dict(lr_mul=lr_mlp, gain=1.),
                    act_cfg=dict(type='fused_bias')))

        self.style_mapping = nn.Sequential(*mapping_layers)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        # constant input layer
        self.constant_input = ConstantInput(self.channels[4])
        # 4x4 stage
        self.conv1 = ModulatedStyleConv(
            self.channels[4],
            self.channels[4],
            kernel_size=3,
            style_channels=style_channels,
            blur_kernel=blur_kernel,
            fp16_enabled=fp16_enabled)
        self.to_rgb1 = ModulatedToRGB(
            self.channels[4],
            style_channels,
            out_channels=out_channels,
            upsample=False,
            fp16_enabled=fp16_enabled)

        # generator backbone (8x8 --> higher resolutions)
        self.log_size = int(np.log2(self.out_size))

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()

        blk_in_channels_ = self.channels[4]  # in channels of the conv blocks

        for i in range(3, self.log_size + 1):
            blk_out_channels_ = self.channels[2**i]

            # If `fp16_enabled` is True, all of layers will be run in auto
            # FP16. In the case of `num_fp16_sacles` > 0, only partial
            # layers will be run in fp16.
            _use_fp16 = (self.log_size - i) < num_fp16_scales or fp16_enabled

            self.convs.append(
                ModulatedStyleConv(
                    blk_in_channels_,
                    blk_out_channels_,
                    3,
                    style_channels,
                    upsample=True,
                    blur_kernel=blur_kernel,
                    fp16_enabled=_use_fp16))
            self.convs.append(
                ModulatedStyleConv(
                    blk_out_channels_,
                    blk_out_channels_,
                    3,
                    style_channels,
                    upsample=False,
                    blur_kernel=blur_kernel,
                    fp16_enabled=_use_fp16))
            self.to_rgbs.append(
                ModulatedToRGB(
                    blk_out_channels_,
                    style_channels,
                    out_channels=out_channels,
                    upsample=True,
                    fp16_enabled=_use_fp16))  # set to global fp16

            blk_in_channels_ = blk_out_channels_

        self.num_latents = self.log_size * 2 - 2
        self.num_injected_noises = self.num_latents - 1

        # register buffer for injected noises
        for layer_idx in range(self.num_injected_noises):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2**res, 2**res]
            self.register_buffer(f'injected_noise_{layer_idx}',
                                 torch.randn(*shape))

        if (self.cond_size is not None
                and self.cond_size > 0) or update_mean_latent_with_ema:
            # Due to `get_mean_latent` cannot handle conditional input,
            # assign avg style code here and update with EMA.
            self.register_buffer('w_avg', torch.zeros([style_channels]))
            self.w_avg_beta = w_avg_beta
            mmengine.print_log('Mean latent code (w) is updated with EMA.')

        if pretrained is not None:
            self._load_pretrained_model(**pretrained)

    def _load_pretrained_model(self,
                               ckpt_path,
                               prefix='',
                               map_location='cpu',
                               strict=True):
        state_dict = _load_checkpoint_with_prefix(prefix, ckpt_path,
                                                  map_location)
        self.load_state_dict(state_dict, strict=strict)
        mmengine.print_log(f'Load pretrained model from {ckpt_path}')

    def train(self, mode=True):
        if mode:
            if self.default_style_mode != self._default_style_mode:
                mmengine.print_log(
                    f'Switch to train style mode: {self._default_style_mode}')
            self.default_style_mode = self._default_style_mode

        else:
            if self.default_style_mode != self.eval_style_mode:
                mmengine.print_log(
                    f'Switch to evaluation style mode: {self.eval_style_mode}')
            self.default_style_mode = self.eval_style_mode

        return super(StyleGAN2Generator, self).train(mode)

    def make_injected_noise(self):
        """make noises that will be injected into feature maps.

        Returns:
            list[Tensor]: List of layer-wise noise tensor.
        """
        device = get_module_device(self)

        noises = [torch.randn(1, 1, 2**2, 2**2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2**i, 2**i, device=device))

        return noises

    def get_mean_latent(self, num_samples=4096, **kwargs):
        """Get mean latent of W space in this generator.

        Args:
            num_samples (int, optional): Number of sample times. Defaults
                to 4096.

        Returns:
            Tensor: Mean latent of this generator.
        """
        if hasattr(self, 'w_avg'):
            mmengine.print_log('Get latent code (w) which is updated by EMA.')
            return self.w_avg
        return get_mean_latent(self, num_samples, **kwargs)

    def style_mixing(self,
                     n_source,
                     n_target,
                     inject_index=1,
                     truncation_latent=None,
                     truncation=0.7):
        return style_mixing(
            self,
            n_source=n_source,
            n_target=n_target,
            inject_index=inject_index,
            truncation=truncation,
            truncation_latent=truncation_latent,
            style_channels=self.style_channels)

    # @auto_fp16()
    def forward(self,
                styles,
                label=None,
                num_batches=-1,
                return_noise=False,
                return_latents=False,
                inject_index=None,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                injected_noise=None,
                add_noise=True,
                randomize_noise=True,
                update_ws=False):
        """Forward function.

        This function has been integrated with the truncation trick. Please
        refer to the usage of `truncation` and `truncation_latent`.

        Args:
            styles (torch.Tensor | list[torch.Tensor] | callable | None): In
                StyleGAN2, you can provide noise tensor or latent tensor. Given
                a list containing more than one noise or latent tensors, style
                mixing trick will be used in training. Of course, You can
                directly give a batch of noise through a ``torch.Tensor`` or
                offer a callable function to sample a batch of noise data.
                Otherwise, the ``None`` indicates to use the default noise
                sampler.
            label (torch.Tensor, optional): Conditional inputs for the
                generator. Defaults to None.
            num_batches (int, optional): The number of batch size.
                Defaults to 0.
            return_noise (bool, optional): If True, ``noise_batch`` will be
                returned in a dict with ``fake_img``. Defaults to False.
            return_latents (bool, optional): If True, ``latent`` will be
                returned in a dict with ``fake_img``. Defaults to False.
            inject_index (int | None, optional): The index number for mixing
                style codes. Defaults to None.
            truncation (float, optional): Truncation factor. Give value less
                than 1., the truncation trick will be adopted. Defaults to 1.
            truncation_latent (torch.Tensor, optional): Mean truncation latent.
                Defaults to None.
            input_is_latent (bool, optional): If `True`, the input tensor is
                the latent tensor. Defaults to False.
            injected_noise (torch.Tensor | None, optional): Given a tensor, the
                random noise will be fixed as this input injected noise.
                Defaults to None.
            add_noise (bool): Whether apply noise injection. Defaults to True.
            randomize_noise (bool, optional): If `False`, images are sampled
                with the buffered noise tensor injected to the style conv
                block. Defaults to True.
            update_ws (bool): Whether update latent code with EMA. Only work
                when `w_avg` is registeried. Defaults to False.

        Returns:
            torch.Tensor | dict: Generated image tensor or dictionary \
                containing more data.
        """
        input_dim = self.style_channels if input_is_latent else self.noise_size
        # receive noise and conduct sanity check.
        if isinstance(styles, torch.Tensor):
            assert styles.shape[1] == input_dim
            styles = [styles]
        elif mmengine.is_seq_of(styles, torch.Tensor):
            for t in styles:
                assert t.shape[-1] == input_dim
        # receive a noise generator and sample noise.
        elif callable(styles):
            device = get_module_device(self)
            noise_generator = styles
            assert num_batches > 0
            if self.default_style_mode == 'mix' and random.random(
            ) < self.mix_prob:
                styles = [
                    noise_generator((num_batches, input_dim)) for _ in range(2)
                ]
            else:
                styles = [noise_generator((num_batches, input_dim))]
            styles = [s.to(device) for s in styles]
        # otherwise, we will adopt default noise sampler.
        else:
            device = get_module_device(self)
            assert num_batches > 0 and not input_is_latent
            if self.default_style_mode == 'mix' and random.random(
            ) < self.mix_prob:
                styles = [
                    torch.randn((num_batches, input_dim)) for _ in range(2)
                ]
            else:
                styles = [torch.randn((num_batches, input_dim))]
            styles = [s.to(device) for s in styles]

        # no amp for style-mapping and condition-embedding
        if not input_is_latent:
            noise_batch = styles
            if self.cond_size is not None and self.cond_size > 0:
                assert label is not None, (
                    '\'cond_channels\' is not None, \'cond\' must be passed.')
                assert label.shape[1] == self.cond_size
                embedding = self.embed(label)
                # NOTE: If conditional input is passed, do norm for cond
                # embedding and noise input respectively
                # do pixel_norm (2nd_momuent_norm) to cond embedding
                embedding = self.pixel_norm(embedding)
                # do pixel_norm (2nd_momuent_norm) to noise input
                styles = [self.pixel_norm(s) for s in styles]

            styles_list = []
            for s in styles:
                if self.cond_size is not None and self.cond_size > 0:
                    s = torch.cat([s, embedding], dim=1)
                styles_list.append(self.style_mapping(s))

            styles = styles_list
        else:
            noise_batch = None

        # update w_avg during training, if need
        if hasattr(self, 'w_avg') and self.training and update_ws:
            # only update w_avg with the first style code
            self.w_avg.copy_(styles[0].detach().mean(
                dim=0).lerp(self.w_avg, self.w_avg_beta))

        if injected_noise is None:
            if randomize_noise:
                injected_noise = [None] * self.num_injected_noises
            else:
                injected_noise = [
                    getattr(self, f'injected_noise_{i}')
                    for i in range(self.num_injected_noises)
                ]
        # use truncation trick
        if truncation < 1:
            style_t = []
            # calculate truncation latent on the fly
            if truncation_latent is None and not hasattr(
                    self, 'truncation_latent'):
                self.truncation_latent = self.get_mean_latent()
                truncation_latent = self.truncation_latent
            elif truncation_latent is None and hasattr(self,
                                                       'truncation_latent'):
                truncation_latent = self.truncation_latent

            for style in styles:
                style_t.append(truncation_latent + truncation *
                               (style - truncation_latent))

            styles = style_t
        # no style mixing
        if len(styles) < 2:
            inject_index = self.num_latents

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

            else:
                latent = styles[0]
        # style mixing
        else:
            if inject_index is None:
                inject_index = random.randint(1, self.num_latents - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(
                1, self.num_latents - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        with autocast(enabled=self.fp16_enabled):
            # 4x4 stage
            out = self.constant_input(latent)
            if self.fp16_enabled:
                out = out.to(torch.float16)
            out = self.conv1(
                out,
                latent[:, 0],
                noise=injected_noise[0],
                add_noise=add_noise)
            skip = self.to_rgb1(out, latent[:, 1])

            _index = 1

            # 8x8 ---> higher resolutions
            for up_conv, conv, noise1, noise2, to_rgb in zip(
                    self.convs[::2], self.convs[1::2], injected_noise[1::2],
                    injected_noise[2::2], self.to_rgbs):
                out = up_conv(
                    out, latent[:, _index], noise=noise1, add_noise=add_noise)
                out = conv(
                    out,
                    latent[:, _index + 1],
                    noise=noise2,
                    add_noise=add_noise)
                skip = to_rgb(out, latent[:, _index + 2], skip)
                _index += 2

        # make sure the output image is torch.float32 to avoid RunTime Error
        # in other modules
        img = skip.to(torch.float32)

        if self.bgr2rgb:
            img = torch.flip(img, dims=1)

        if return_latents or return_noise:
            output_dict = dict(
                fake_img=img,
                latent=latent,
                inject_index=inject_index,
                noise_batch=noise_batch)
            return output_dict

        return img
