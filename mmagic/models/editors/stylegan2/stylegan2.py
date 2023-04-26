# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine import Config, MessageHub
from mmengine.model import BaseModel, is_model_wrapper
from mmengine.optim import OptimWrapper, OptimWrapperDict
from torch import Tensor

from mmagic.registry import MODELS
from mmagic.structures import DataSample
from ...base_models import BaseGAN
from ...losses import gen_path_regularizer, r1_gradient_penalty_loss
from ...utils import set_requires_grad

ModelType = Union[Dict, nn.Module]


@MODELS.register_module()
class StyleGAN2(BaseGAN):
    """Implementation of `Analyzing and Improving the Image Quality of
    Stylegan`. # noqa.

    Paper link: https://openaccess.thecvf.com/content_CVPR_2020/html/Karras_Analyzing_and_Improving_the_Image_Quality_of_StyleGAN_CVPR_2020_paper.html. # noqa

    :class:`~mmagic.models.editors.stylegan2.StyleGAN2Generator`
    and
    :class:`~mmagic.models.editors.stylegan2.StyleGAN2Discriminator`

    Args:
        generator (ModelType): The config or model of the generator.
        discriminator (Optional[ModelType]): The config or model of the
            discriminator. Defaults to None.
        data_preprocessor (Optional[Union[dict, Config]]): The pre-process
            config or :class:`~mmagic.models.DataPreprocessor`.
        generator_steps (int): The number of times the generator is completely
            updated before the discriminator is updated. Defaults to 1.
        discriminator_steps (int): The number of times the discriminator is
            completely updated before the generator is updated. Defaults to 1.
        ema_config (Optional[Dict]): The config for generator's exponential
            moving average setting. Defaults to None.
    """

    def __init__(self,
                 generator: ModelType,
                 discriminator: Optional[ModelType] = None,
                 data_preprocessor: Optional[Union[dict, Config]] = None,
                 generator_steps: int = 1,
                 discriminator_steps: int = 1,
                 ema_config: Optional[Dict] = None,
                 loss_config=dict()):
        BaseModel.__init__(self, data_preprocessor=data_preprocessor)

        # build generator
        if isinstance(generator, dict):
            self._gen_cfg = deepcopy(generator)
            generator = MODELS.build(generator)
        self.generator = generator

        # get valid noise_size
        self.noise_size = getattr(self.generator, 'style_channels', 512)

        # build discriminator
        if discriminator:
            if isinstance(discriminator, dict):
                self._disc_cfg = deepcopy(discriminator)
                # build discriminator with default `num_classes`
                disc_args = dict()
                if hasattr(self, 'num_classes'):
                    disc_args['num_classes'] = self.num_classes
                discriminator = MODELS.build(
                    discriminator, default_args=disc_args)
        self.discriminator = discriminator

        self._gen_steps = generator_steps
        self._disc_steps = discriminator_steps

        if ema_config is None:
            self._ema_config = None
            self._with_ema_gen = False
        else:
            self._ema_config = deepcopy(ema_config)
            self._init_ema_model(self._ema_config)
            self._with_ema_gen = True

        # loss config
        self.loss_config = deepcopy(loss_config)
        # r1 settings
        self.r1_loss_weight = self.loss_config.get('r1_loss_weight', 80.0)
        self.r1_interval = self.loss_config.get('r1_interval', 16)
        self.norm_mode = self.loss_config.get('norm_mode', 'pixel')
        self.r1_use_apex_amp = self.loss_config.get('r1_use_apex_amp', False)
        self.scale_r1_loss = self.loss_config.get('scale_r1_loss', False)
        # gen path reg settings
        self.g_reg_interval = self.loss_config.get('g_reg_interval', 4)
        self.g_reg_weight = self.loss_config.get('g_reg_weight', 8.)
        self.pl_batch_shrink = self.loss_config.get('pl_batch_shrink', 2)
        self.g_reg_use_apex_amp = self.loss_config.get('g_reg_use_apex_amp',
                                                       False)
        self.register_buffer('mean_path_length', torch.tensor(0.))

    def disc_loss(self, disc_pred_fake: Tensor, disc_pred_real: Tensor,
                  real_imgs: Tensor) -> Tuple:
        r"""Get disc loss. StyleGANv2 use the non-saturating loss and R1
            gradient penalty to train the discriminator.

        Args:
            disc_pred_fake (Tensor): Discriminator's prediction of the fake
                images.
            disc_pred_real (Tensor): Discriminator's prediction of the real
                images.
            real_imgs (Tensor): Input real images.

        Returns:
            tuple[Tensor, dict]: Loss value and a dict of log variables.
        """

        losses_dict = dict()
        # no-saturating gan loss
        losses_dict['loss_disc_fake'] = F.softplus(disc_pred_fake).mean()
        losses_dict['loss_disc_real'] = F.softplus(-disc_pred_real).mean()
        # R1 Gradient Penalty
        message_hub = MessageHub.get_current_instance()
        curr_iter = message_hub.get_info('iter')
        if curr_iter % self.r1_interval == 0:
            losses_dict[
                'loss_r1_gp'] = self.r1_loss_weight * r1_gradient_penalty_loss(
                    self.discriminator,
                    real_imgs,
                    norm_mode=self.norm_mode,
                    use_apex_amp=self.r1_use_apex_amp)

        loss, log_var = self.parse_losses(losses_dict)
        return loss, log_var

    def gen_loss(self, disc_pred_fake: Tensor, batch_size: int) -> Tuple:
        """Get gen loss. StyleGANv2 use the non-saturating loss and generator
        path regularization to train the generator.

        Args:
            disc_pred_fake (Tensor): Discriminator's prediction of the fake
                images.
            batch_size (int): Batch size for generating fake images.

        Returns:
            tuple[Tensor, dict]: Loss value and a dict of log variables.
        """
        losses_dict = dict()
        # no-saturating gan loss
        losses_dict['loss_gen'] = F.softplus(-disc_pred_fake).mean()

        # Generator Path Regularizer
        message_hub = MessageHub.get_current_instance()
        curr_iter = message_hub.get_info('iter')
        if curr_iter % self.g_reg_interval == 0:
            path_penalty, self.mean_path_length, _ = gen_path_regularizer(
                self.generator,
                batch_size,
                self.mean_path_length,
                pl_batch_shrink=self.pl_batch_shrink,
                use_apex_amp=self.g_reg_use_apex_amp)
            losses_dict['loss_path_regular'] = self.g_reg_weight * path_penalty
        loss, log_var = self.parse_losses(losses_dict)
        return loss, log_var

    def train_discriminator(self, inputs: dict, data_samples: DataSample,
                            optimizer_wrapper: OptimWrapper
                            ) -> Dict[str, Tensor]:
        """Train discriminator.

        Args:
            inputs (dict): Inputs from dataloader.
            data_samples (DataSample): Data samples from dataloader.
            optim_wrapper (OptimWrapper): OptimWrapper instance used to update
                model parameters.
        Returns:
            Dict[str, Tensor]: A ``dict`` of tensor for logging.
        """
        real_imgs = data_samples.gt_img

        num_batches = real_imgs.shape[0]

        noise_batch = self.noise_fn(num_batches=num_batches)
        with torch.no_grad():
            fake_imgs = self.generator(noise_batch, return_noise=False)

        disc_pred_fake = self.discriminator(fake_imgs)
        disc_pred_real = self.discriminator(real_imgs)

        parsed_losses, log_vars = self.disc_loss(disc_pred_fake,
                                                 disc_pred_real, real_imgs)
        optimizer_wrapper.update_params(parsed_losses)
        # save ada info
        message_hub = MessageHub.get_current_instance()
        message_hub.update_info('disc_pred_real', disc_pred_real)
        return log_vars

    def train_generator(self, inputs: dict, data_samples: DataSample,
                        optimizer_wrapper: OptimWrapper) -> Dict[str, Tensor]:
        """Train generator.

        Args:
            inputs (dict): Inputs from dataloader.
            data_samples (DataSample): Data samples from dataloader.
                Do not used in generator's training.
            optim_wrapper (OptimWrapper): OptimWrapper instance used to update
                model parameters.

        Returns:
            Dict[str, Tensor]: A ``dict`` of tensor for logging.
        """
        num_batches = len(data_samples)

        noise = self.noise_fn(num_batches=num_batches)
        fake_imgs = self.generator(noise, return_noise=False)

        disc_pred_fake = self.discriminator(fake_imgs)
        parsed_loss, log_vars = self.gen_loss(disc_pred_fake, num_batches)

        optimizer_wrapper.update_params(parsed_loss)
        return log_vars

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
        inputs_dict, data_samples = data['inputs'], data['data_samples']

        disc_optimizer_wrapper: OptimWrapper = optim_wrapper['discriminator']
        disc_accu_iters = disc_optimizer_wrapper._accumulative_counts

        # NOTE: Do not use context manager of optim_wrapper. Because
        # in mixed-precision training, StyleGAN2 only enable fp16 in
        # specified blocks (refers to `:attr:enable_fp16` in
        # :class:`~StyleGANv2Generator` and :class:`~StyleGAN2Discriminator`
        # for more details), but in :func:`~AmpOptimWrapper.optim_context`,
        # fp16 is applied to all modules. This may slow down gradient
        # accumulation because `no_sycn` in
        # :func:`~OptimWrapper.optim_context` will not be called any more.
        log_vars = self.train_discriminator(inputs_dict, data_samples,
                                            disc_optimizer_wrapper)

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
                log_vars_gen = self.train_generator(inputs_dict, data_samples,
                                                    gen_optimizer_wrapper)

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

        batch_size = len(data['data_samples'])
        # update ada p
        if hasattr(self.discriminator,
                   'with_ada') and self.discriminator.with_ada:
            self.discriminator.ada_aug.log_buffer[0] += batch_size
            self.discriminator.ada_aug.log_buffer[1] += message_hub.get_info(
                'disc_pred_real').sign().sum()

            self.discriminator.ada_aug.update(
                iteration=curr_iter, num_batches=batch_size)
            log_vars['augment'] = (
                self.discriminator.ada_aug.aug_pipeline.p.data.cpu())

        return log_vars
