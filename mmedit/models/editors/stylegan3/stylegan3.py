# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from mmengine import Config, MessageHub
from mmengine.optim import OptimWrapper
from torch import Tensor

from mmedit.registry import MODELS
from mmedit.structures import EditDataSample
from mmedit.utils.typing import SampleList
from ...utils import get_module_device, get_valid_num_batches
from ..stylegan2 import StyleGAN2
from .stylegan3_utils import (apply_fractional_pseudo_rotation,
                              apply_fractional_rotation,
                              apply_fractional_translation,
                              apply_integer_translation, rotation_matrix)

ModelType = Union[Dict, nn.Module]


@MODELS.register_module()
class StyleGAN3(StyleGAN2):
    """Impelmentation of `Alias-Free Generative Adversarial Networks`. # noqa.

    Paper link: https://nvlabs-fi-cdn.nvidia.com/stylegan3/stylegan3-paper.pdf # noqa

    Detailed architecture can be found in

    :class:~`mmgen.models.architectures.stylegan.generator_discriminator_v3.StyleGANv3Generator`  # noqa
    and
    :class:~`mmgen.models.architectures.stylegan.generator_discriminator_v2.StyleGAN2Discriminator`  # noqa
    """

    def __init__(self,
                 generator: ModelType,
                 discriminator: Optional[ModelType] = None,
                 data_preprocessor: Optional[Union[dict, Config]] = None,
                 generator_steps: int = 1,
                 discriminator_steps: int = 1,
                 forward_kwargs: Optional[Dict] = None,
                 ema_config: Optional[Dict] = None,
                 loss_config=dict()):
        super().__init__(generator, discriminator, data_preprocessor,
                         generator_steps, discriminator_steps, ema_config,
                         loss_config)

        self.noise_size = getattr(self.generator, 'noise_size', 512)
        forward_kwargs = dict() if forward_kwargs is None else forward_kwargs
        disc_default_forward_kwargs = dict(update_emas=True, force_fp32=False)
        gen_default_forward_kwargs = dict(force_fp32=False)
        forward_kwargs.setdefault('disc', disc_default_forward_kwargs)
        forward_kwargs.setdefault('gen', gen_default_forward_kwargs)
        self.forward_kwargs = forward_kwargs

    def test_step(self, data: dict) -> SampleList:
        """Gets the generated image of given data. Same as :meth:`val_step`.

        Args:
            data (dict): Data sampled from metric specific
                sampler. More detials in `Metrics` and `Evaluator`.

        Returns:
            SampleList: A list of ``EditDataSample`` contain generated results.
        """
        data = self.data_preprocessor(data)
        inputs_dict, data_samples = data['inputs'], data['data_samples']
        # hard code to compute equivarience
        if 'mode' in inputs_dict and 'eq_cfg' in inputs_dict['mode']:
            batch_size = get_valid_num_batches(inputs_dict)
            outputs = self.sample_equivarience_pairs(
                batch_size,
                sample_mode=inputs_dict['mode']['sample_mode'],
                eq_cfg=inputs_dict['mode']['eq_cfg'],
                sample_kwargs=inputs_dict['mode']['sample_kwargs'])
        else:
            outputs = self(inputs_dict, data_samples)
        return outputs

    def val_step(self, data: dict) -> SampleList:
        """Gets the generated image of given data. Same as :meth:`val_step`.

        Args:
            data (dict): Data sampled from metric specific
                sampler. More detials in `Metrics` and `Evaluator`.

        Returns:
            SampleList: A list of ``EditDataSample`` contain generated results.
        """
        data = self.data_preprocessor(data)
        inputs_dict, data_samples = data['inputs'], data['data_samples']
        # hard code to compute equivarience
        if 'mode' in inputs_dict and 'eq_cfg' in inputs_dict['mode']:
            batch_size = get_valid_num_batches(inputs_dict)
            outputs = self.sample_equivarience_pairs(
                batch_size,
                sample_mode=inputs_dict['mode']['sample_mode'],
                eq_cfg=inputs_dict['mode']['eq_cfg'],
                sample_kwargs=inputs_dict['mode']['sample_kwargs'])
        else:
            outputs = self(inputs_dict, data_samples)
        return outputs

    def train_discriminator(self, inputs: dict,
                            data_samples: List[EditDataSample],
                            optimizer_wrapper: OptimWrapper
                            ) -> Dict[str, Tensor]:
        """Train discriminator.

        Args:
            inputs (dict): Inputs from dataloader.
            data_samples (List[EditDataSample]): Data samples from dataloader.
            optim_wrapper (OptimWrapper): OptimWrapper instance used to update
                model parameters.
        Returns:
            Dict[str, Tensor]: A ``dict`` of tensor for logging.
        """
        real_imgs = inputs['img']

        num_batches = real_imgs.shape[0]

        noise_batch = self.noise_fn(num_batches=num_batches)
        with torch.no_grad():
            fake_imgs = self.generator(
                noise_batch, return_noise=False, **self.forward_kwargs['disc'])

        disc_pred_fake = self.discriminator(fake_imgs)
        disc_pred_real = self.discriminator(real_imgs)

        parsed_losses, log_vars = self.disc_loss(disc_pred_fake,
                                                 disc_pred_real, real_imgs)
        optimizer_wrapper.update_params(parsed_losses)
        # save ada info
        message_hub = MessageHub.get_current_instance()
        message_hub.update_info('disc_pred_real', disc_pred_real)
        return log_vars

    def train_generator(self, inputs: dict, data_samples: List[EditDataSample],
                        optimizer_wrapper: OptimWrapper) -> Dict[str, Tensor]:
        """Train generator.

        Args:
            inputs (dict): Inputs from dataloader.
            data_samples (List[EditDataSample]): Data samples from dataloader.
                Do not used in generator's training.
            optim_wrapper (OptimWrapper): OptimWrapper instance used to update
                model parameters.

        Returns:
            Dict[str, Tensor]: A ``dict`` of tensor for logging.
        """
        num_batches = inputs['img'].shape[0]

        noise = self.noise_fn(num_batches=num_batches)
        fake_imgs = self.generator(
            noise, return_noise=False, **self.forward_kwargs['gen'])

        disc_pred_fake = self.discriminator(fake_imgs)
        parsed_loss, log_vars = self.gen_loss(disc_pred_fake, num_batches)

        optimizer_wrapper.update_params(parsed_loss)
        return log_vars

    def sample_equivarience_pairs(self,
                                  batch_size,
                                  sample_mode='ema',
                                  eq_cfg=dict(
                                      compute_eqt_int=False,
                                      compute_eqt_frac=False,
                                      compute_eqr=False,
                                      translate_max=0.125,
                                      rotate_max=1),
                                  sample_kwargs=dict()):
        generator = self.generator if (sample_mode
                                       == 'orig') else self.generator_ema
        if hasattr(generator, 'module'):
            generator = generator.module

        device = get_module_device(generator)
        identity_matrix = torch.eye(3, device=device)

        # Run mapping network.
        z = torch.randn([batch_size, self.noise_size], device=device)
        ws = generator.style_mapping(z=z)
        transform_matrix = getattr(
            getattr(getattr(generator, 'synthesis', None), 'input', None),
            'transform', None)

        # Generate reference image.
        transform_matrix[:] = identity_matrix
        orig = generator.synthesis(ws=ws, **sample_kwargs)

        batch_sample = [EditDataSample() for _ in range(batch_size)]
        # Integer translation (EQ-T).
        if eq_cfg['compute_eqt_int']:
            t = (torch.rand(2, device=device) * 2 -
                 1) * eq_cfg['translate_max']
            t = (t * generator.out_size).round() / generator.out_size
            transform_matrix[:] = identity_matrix
            transform_matrix[:2, 2] = -t
            img = generator.synthesis(ws=ws, **sample_kwargs)
            ref, mask = apply_integer_translation(orig, t[0], t[1])

            diff = (ref - img).square() * mask
            for idx in range(batch_size):
                data_sample = batch_sample[idx]
                setattr(data_sample, 'eqt_int',
                        EditDataSample(diff=diff, mask=mask))

        # Fractional translation (EQ-T_frac).
        if eq_cfg['compute_eqt_frac']:
            t = (torch.rand(2, device=device) * 2 -
                 1) * eq_cfg['translate_max']
            transform_matrix[:] = identity_matrix
            transform_matrix[:2, 2] = -t
            img = generator.synthesis(ws=ws, **sample_kwargs)
            ref, mask = apply_fractional_translation(orig, t[0], t[1])

            diff = (ref - img).square() * mask
            for idx in range(batch_size):
                data_sample = batch_sample[idx]
                setattr(data_sample, 'eqt_frac',
                        EditDataSample(diff=diff, mask=mask))

        # Rotation (EQ-R).
        if eq_cfg['compute_eqr']:
            angle = (torch.rand([], device=device) * 2 - 1) * (
                eq_cfg['rotate_max'] * np.pi)
            transform_matrix[:] = rotation_matrix(-angle)
            img = generator.synthesis(ws=ws, **sample_kwargs)
            ref, ref_mask = apply_fractional_rotation(orig, angle)
            pseudo, pseudo_mask = apply_fractional_pseudo_rotation(img, angle)
            mask = ref_mask * pseudo_mask

            diff = (ref - pseudo).square() * mask
            for idx in range(batch_size):
                data_sample = batch_sample[idx]
                setattr(data_sample, 'eqr',
                        EditDataSample(diff=diff, mask=mask))

        return batch_sample
