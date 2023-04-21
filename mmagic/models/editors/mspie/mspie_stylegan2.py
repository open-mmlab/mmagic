# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from typing import Dict, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from mmengine import MessageHub
from mmengine.logging import MMLogger
from mmengine.model import is_model_wrapper
from mmengine.optim import OptimWrapper, OptimWrapperDict
from torch import Tensor

from mmagic.registry import MODELS
from mmagic.structures import DataSample
from ...utils import set_requires_grad
from ..stylegan2 import StyleGAN2

ModelType = Union[Dict, nn.Module]
TrainInput = Union[dict, Tensor]


@MODELS.register_module()
class MSPIEStyleGAN2(StyleGAN2):
    """MS-PIE StyleGAN2.

    In this GAN, we adopt the MS-PIE training schedule so that multi-scale
    images can be generated with a single generator. Details can be found in:
    Positional Encoding as Spatial Inductive Bias in GANs, CVPR2021.

    Args:
        train_settings (dict): Config for training settings.
            Defaults to `dict()`.
    """

    def __init__(self, *args, train_settings=dict(), **kwargs):
        super().__init__(*args, **kwargs)
        self.train_settings = deepcopy(train_settings)
        # set the number of upsampling blocks. This value will be used to
        # calculate the current result size according to the size of the input
        # feature map, e.g., positional encoding map
        self.num_upblocks = self.train_settings.get('num_upblocks', 6)

        # multiple input scales (a list of int) that will be added to the
        # original starting scale.
        self.multi_input_scales = self.train_settings.get('multi_input_scales')
        self.multi_scale_probability = self.train_settings.get(
            'multi_scale_probability')

    def train_step(self, data: dict,
                   optim_wrapper: OptimWrapperDict) -> Dict[str, Tensor]:
        """Train GAN model. In the training of GAN models, generator and
        discriminator are updated alternatively. In MMagic's design,
        `self.train_step` is called with data input. Therefore we always update
        discriminator, whose updating is relay on real data, and then determine
        if the generator needs to be updated based on the current number of
        iterations. More details about whether to update generator can be found
        in :meth:`should_gen_update`.

        Args:
            data (dict): Data sampled from dataloader.
            optim_wrapper (OptimWrapperDict): OptimWrapperDict instance
                contains OptimWrapper of generator and discriminator.

        Returns:
            Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
        """
        message_hub = MessageHub.get_current_instance()
        curr_iter = message_hub.get_info('iter')
        data = self.data_preprocessor(data, True)

        disc_optimizer_wrapper: OptimWrapper = optim_wrapper['discriminator']
        disc_accu_iters = disc_optimizer_wrapper._accumulative_counts

        with disc_optimizer_wrapper.optim_context(self.discriminator):
            log_vars = self.train_discriminator(
                **data, optimizer_wrapper=disc_optimizer_wrapper)

        # add 1 to `curr_iter` because iter is updated in train loop.
        # Whether to update the generator. We update generator with
        # discriminator is fully updated for `self.n_discriminator_steps`
        # iterations. And one full updating for discriminator contains
        # `disc_accu_counts` times of grad accumulations.
        if (curr_iter + 1) % (self.discriminator_steps * disc_accu_iters) == 0:
            set_requires_grad(self.discriminator, False)
            gen_optimizer_wrapper = optim_wrapper['generator']
            gen_accu_iters = gen_optimizer_wrapper._accumulative_counts

            log_vars_gen_list = []
            # init optimizer wrapper status for generator manually
            gen_optimizer_wrapper.initialize_count_status(
                self.generator, 0, self.generator_steps * gen_accu_iters)
            for _ in range(self.generator_steps * gen_accu_iters):
                with gen_optimizer_wrapper.optim_context(self.generator):
                    log_vars_gen = self.train_generator(
                        **data, optimizer_wrapper=gen_optimizer_wrapper)

                log_vars_gen_list.append(log_vars_gen)
            log_vars_gen = self.gather_log_vars(log_vars_gen_list)
            log_vars_gen.pop('loss', None)  # remove 'loss' from gen logs

            set_requires_grad(self.discriminator, True)

            # only do ema after generator update
            if self.with_ema_gen and (curr_iter + 1) >= (
                    self.ema_start * self.discriminator_steps *
                    disc_accu_iters):
                self.generator_ema.update_parameters(
                    self.generator.module
                    if is_model_wrapper(self.generator) else self.generator)
                # if not update buffer, copy buffer from orig model
                if not self.generator_ema.update_buffers:
                    self.generator_ema.sync_buffers(
                        self.generator.module if is_model_wrapper(
                            self.generator) else self.generator)
            elif self.with_ema_gen:
                # before ema, copy weights from orig
                self.generator_ema.sync_parameters(
                    self.generator.module
                    if is_model_wrapper(self.generator) else self.generator)

            log_vars.update(log_vars_gen)

        return log_vars

    def train_generator(self, inputs: dict, data_samples: DataSample,
                        optimizer_wrapper: OptimWrapper) -> Dict[str, Tensor]:
        """Train generator.

        Args:
            inputs (TrainInput): Inputs from dataloader.
            data_samples (DataSample): Data samples from dataloader.
                Do not used in generator's training.
            optim_wrapper (OptimWrapper): OptimWrapper instance used to update
                model parameters.

        Returns:
            Dict[str, Tensor]: A ``dict`` of tensor for logging.
        """
        num_batches = len(data_samples)

        noise = self.noise_fn(num_batches=num_batches)
        fake_imgs = self.generator(
            noise, return_noise=False, chosen_scale=self.chosen_scale)

        disc_pred_fake = self.discriminator(fake_imgs)
        parsed_loss, log_vars = self.gen_loss(disc_pred_fake, num_batches)

        optimizer_wrapper.update_params(parsed_loss)
        return log_vars

    def train_discriminator(self, inputs: dict, data_samples: DataSample,
                            optimizer_wrapper: OptimWrapper
                            ) -> Dict[str, Tensor]:
        """Train discriminator.

        Args:
            inputs (TrainInput): Inputs from dataloader.
            data_samples (DataSample): Data samples from dataloader.
            optim_wrapper (OptimWrapper): OptimWrapper instance used to update
                model parameters.
        Returns:
            Dict[str, Tensor]: A ``dict`` of tensor for logging.
        """
        real_imgs = data_samples.gt_img

        if dist.is_initialized():
            # randomly sample a scale for current training iteration
            chosen_scale = np.random.choice(self.multi_input_scales, 1,
                                            self.multi_scale_probability)[0]

            chosen_scale = torch.tensor(chosen_scale, dtype=torch.int).cuda()
            dist.broadcast(chosen_scale, 0)
            chosen_scale = int(chosen_scale.item())

        else:
            logger = MMLogger.get_current_instance()
            logger.info(
                'Distributed training has not been initialized. Degrade to '
                'the standard stylegan2')
            chosen_scale = 0

        curr_size = (4 + chosen_scale) * (2**self.num_upblocks)
        # adjust the shape of images
        if real_imgs.shape[-2:] != (curr_size, curr_size):
            real_imgs = F.interpolate(
                real_imgs,
                size=(curr_size, curr_size),
                mode='bilinear',
                align_corners=True)

        num_batches = real_imgs.shape[0]

        noise_batch = self.noise_fn(num_batches=num_batches)
        with torch.no_grad():
            fake_imgs = self.generator(
                noise_batch, return_noise=False, chosen_scale=chosen_scale)
        # store chosen scale for training generator
        setattr(self, 'chosen_scale', chosen_scale)

        disc_pred_fake = self.discriminator(fake_imgs)
        disc_pred_real = self.discriminator(real_imgs)

        parsed_losses, log_vars = self.disc_loss(disc_pred_fake,
                                                 disc_pred_real, real_imgs)
        optimizer_wrapper.update_params(parsed_losses)
        return log_vars
