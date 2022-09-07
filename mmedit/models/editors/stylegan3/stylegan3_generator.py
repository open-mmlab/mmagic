# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import mmengine
import torch
import torch.nn as nn
from mmengine.runner.checkpoint import _load_checkpoint_with_prefix

from mmedit.registry import MODULES
from ...utils import get_module_device
from ..stylegan1 import get_mean_latent


@MODULES.register_module('StyleGANv3Generator')
@MODULES.register_module()
class StyleGAN3Generator(nn.Module):
    """StyleGAN3 Generator.

    In StyleGAN3, we make several changes to StyleGANv2's generator which
    include transformed fourier features, filtered nonlinearities and
    non-critical sampling, etc. More details can be found in: Alias-Free
    Generative Adversarial Networks NeurIPS'2021.

    Ref: https://github.com/NVlabs/stylegan3

    Args:
        out_size (int): The output size of the StyleGAN3 generator.
        style_channels (int): The number of channels for style code.
        img_channels (int): The number of output's channels.
        noise_size (int, optional): Size of the input noise vector.
            Defaults to 512.
        rgb2bgr (bool, optional): Whether to reformat the output channels
                with order `bgr`. We provide several pre-trained StyleGAN3
                weights whose output channels order is `rgb`. You can set
                this argument to True to use the weights.
        pretrained (str | dict, optional): Path for the pretrained model or
            dict containing information for pretained models whose necessary
            key is 'ckpt_path'. Besides, you can also provide 'prefix' to load
            the generator part from the whole state dict. Defaults to None.
        synthesis_cfg (dict, optional): Config for synthesis network. Defaults
            to dict(type='SynthesisNetwork').
        mapping_cfg (dict, optional): Config for mapping network. Defaults to
            dict(type='MappingNetwork').
    """

    def __init__(self,
                 out_size,
                 style_channels,
                 img_channels,
                 noise_size=512,
                 rgb2bgr=False,
                 pretrained=None,
                 synthesis_cfg=dict(type='SynthesisNetwork'),
                 mapping_cfg=dict(type='MappingNetwork')):
        super().__init__()
        self.noise_size = noise_size
        self.style_channels = style_channels
        self.out_size = out_size
        self.img_channels = img_channels
        self.rgb2bgr = rgb2bgr

        self._synthesis_cfg = deepcopy(synthesis_cfg)
        self._synthesis_cfg.setdefault('style_channels', style_channels)
        self._synthesis_cfg.setdefault('out_size', out_size)
        self._synthesis_cfg.setdefault('img_channels', img_channels)
        self.synthesis = MODULES.build(self._synthesis_cfg)

        self.num_ws = self.synthesis.num_ws
        self._mapping_cfg = deepcopy(mapping_cfg)
        self._mapping_cfg.setdefault('noise_size', noise_size)
        self._mapping_cfg.setdefault('style_channels', style_channels)
        self._mapping_cfg.setdefault('num_ws', self.num_ws)
        self.style_mapping = MODULES.build(self._mapping_cfg)

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

    def forward(self,
                noise,
                num_batches=0,
                input_is_latent=False,
                truncation=1,
                num_truncation_layer=None,
                update_emas=False,
                force_fp32=True,
                return_noise=False,
                return_latents=False):
        """Forward Function for stylegan3.

        Args:
            noise (torch.Tensor | callable | None): You can directly give a
                batch of noise through a ``torch.Tensor`` or offer a callable
                function to sample a batch of noise data. Otherwise, the
                ``None`` indicates to use the default noise sampler.
            num_batches (int, optional): The number of batch size.
                Defaults to 0.
            input_is_latent (bool, optional): If `True`, the input tensor is
                the latent tensor. Defaults to False.
            truncation (float, optional): Truncation factor. Give value less
                than 1., the truncation trick will be adopted. Defaults to 1.
            num_truncation_layer (int, optional): Number of layers use
                truncated latent. Defaults to None.
            update_emas (bool, optional): Whether update moving average of
                mean latent. Defaults to False.
            force_fp32 (bool, optional): Force fp32 ignore the weights.
                Defaults to True.
            return_noise (bool, optional): If True, ``noise_batch`` will be
                returned in a dict with ``fake_img``. Defaults to False.
            return_latents (bool, optional): If True, ``latent`` will be
                returned in a dict with ``fake_img``. Defaults to False.
        Returns:
            torch.Tensor | dict: Generated image tensor or dictionary \
                containing more data.
        """
        # if input is latent, set noise size as the style_channels
        noise_size = (
            self.style_channels if input_is_latent else self.noise_size)

        if isinstance(noise, torch.Tensor):
            assert noise.shape[1] == noise_size
            assert noise.ndim == 2, ('The noise should be in shape of (n, c), '
                                     f'but got {noise.shape}')
            noise_batch = noise

        # receive a noise generator and sample noise.
        elif callable(noise):
            noise_generator = noise
            assert num_batches > 0
            noise_batch = noise_generator((num_batches, noise_size))

        # otherwise, we will adopt default noise sampler.
        else:
            assert num_batches > 0
            noise_batch = torch.randn((num_batches, noise_size))

        device = get_module_device(self)
        noise_batch = noise_batch.to(device)

        if input_is_latent:
            ws = noise_batch.unsqueeze(1).repeat([1, self.num_ws, 1])
        else:
            ws = self.style_mapping(
                noise_batch,
                truncation=truncation,
                num_truncation_layer=num_truncation_layer,
                update_emas=update_emas)
        out_img = self.synthesis(
            ws, update_emas=update_emas, force_fp32=force_fp32)

        if self.rgb2bgr:
            out_img = out_img[:, [2, 1, 0], ...]

        if return_noise or return_latents:
            output = dict(fake_img=out_img, noise_batch=noise_batch, latent=ws)
            return output

        return out_img

    def get_mean_latent(self, num_samples=4096, **kwargs):
        """Get mean latent of W space in this generator.

        Args:
            num_samples (int, optional): Number of sample times. Defaults
                to 4096.

        Returns:
            Tensor: Mean latent of this generator.
        """
        if hasattr(self.style_mapping, 'w_avg'):
            return self.style_mapping.w_avg
        return get_mean_latent(self, num_samples, **kwargs)

    def get_training_kwargs(self, phase):
        """Get training kwargs. In StyleGANv3, we enable fp16, and update
        mangitude ema during training of discriminator. This function is used
        to pass related arguments.

        Args:
            phase (str): Current training phase.

        Returns:
            dict: Training kwargs.
        """
        if phase == 'disc':
            return dict(update_emas=True, force_fp32=False)
        if phase == 'gen':
            return dict(force_fp32=False)
        return {}
