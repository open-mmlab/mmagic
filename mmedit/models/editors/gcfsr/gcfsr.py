# Copyright (c) OpenMMLab. All rights reserved.
import random
from copy import deepcopy
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from mmengine import Config, MessageHub
from mmengine.model import BaseModel, is_model_wrapper
from mmengine.optim import OptimWrapper, OptimWrapperDict
from torch import Tensor

from mmedit.registry import MODELS, MODULES
from ...base_models import BaseGAN
from ...utils import imresize, set_requires_grad

ModelType = Union[Dict, nn.Module]


@MODELS.register_module()
class GCFSRGAN(BaseGAN):
    """"""

    def __init__(self,
                 generator: ModelType,
                 discriminator: Optional[ModelType] = None,
                 data_preprocessor: Optional[Union[dict, Config]] = None,
                 generator_steps: int = 1,
                 discriminator_steps: int = 1,
                 d_reg_interval: int = 16,
                 r1_reg_weight: int = 10,
                 ema_config: Optional[Dict] = None,
                 pixel_loss: Optional[Dict] = None,
                 perceptual_loss: Optional[Dict] = None,
                 gan_loss: Optional[Dict] = None,
                 rescale_list: Optional[List] = None):
        BaseModel.__init__(self, data_preprocessor=data_preprocessor)

        # build generator
        if isinstance(generator, dict):
            self._gen_cfg = deepcopy(generator)
            generator = MODULES.build(generator)
        self.generator = generator
        self.num_style_feat = self._gen_cfg.get('num_style_feat', 512)

        # build discriminator
        if discriminator:
            if isinstance(discriminator, dict):
                self._disc_cfg = deepcopy(discriminator)
                # build discriminator with default `num_classes`
                disc_args = dict()
                if hasattr(self, 'num_classes'):
                    disc_args['num_classes'] = self.num_classes
                discriminator = MODULES.build(
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

        # rescale config
        self.rescale_list = rescale_list
        self.condition_norm = max(rescale_list)

        self.net_d_reg_every = d_reg_interval
        self.r1_reg_weight = r1_reg_weight

        # loss config
        self.pixel_loss = MODELS.build(pixel_loss) if pixel_loss else None
        self.gan_loss = MODELS.build(gan_loss) if gan_loss else None
        self.perceptual_loss = MODELS.build(
            perceptual_loss) if perceptual_loss else None

    def r1_penalty(self, real_pred, real_img):
        """R1 regularization for discriminator. The core idea is to penalize
        the gradient on real data alone: when the generator distribution
        produces the true data distribution and the discriminator is equal to 0
        on the data manifold, the gradient penalty ensures that the
        discriminator cannot create a non-zero gradient orthogonal to the data
        manifold without suffering a loss in the GAN game.

        Ref:
        Eq. 9 in Which training methods for GANs do actually converge.
        """
        grad_real = torch.autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True)[0]
        grad_penalty = grad_real.pow(2).view(grad_real.shape[0],
                                             -1).sum(1).mean()
        return grad_penalty

    def train_step(self, data: dict,
                   optim_wrapper: OptimWrapperDict) -> Dict[str, Tensor]:
        """

        Args:
            data (dict): Data sampled from dataloader.
            optim_wrapper (OptimWrapperDict): OptimWrapperDict instance
                contains OptimWrapper of generator and discriminator.

        Returns:
            Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
        """
        losses_dict = dict()
        message_hub = MessageHub.get_current_instance()
        curr_iter = message_hub.get_info('iter')
        data = self.data_preprocessor(data, True)
        inputs_dict, _ = data['inputs'], data['data_samples']
        gt_imgs = inputs_dict['img']

        disc_optimizer_wrapper: OptimWrapper = optim_wrapper['discriminator']
        gen_optimizer_wrapper: OptimWrapper = optim_wrapper['generator']
        disc_accu_iters = disc_optimizer_wrapper._accumulative_counts

        set_requires_grad(self.discriminator, True)
        disc_optimizer_wrapper.zero_grad()

        scale = random.choice(self.rescale_list)
        input_imgs = imresize(imresize(gt_imgs, 1 / scale), scale)
        conditions = torch.from_numpy(
            np.array([scale / self.condition_norm],
                     dtype=np.float32)).unsqueeze(0).to(gt_imgs.device)

        fake_imgs, _ = self.generator(input_imgs, conditions)
        fake_pred = self.discriminator(fake_imgs.detach())
        real_pred = self.discriminator(gt_imgs)

        loss_disc = self.gan_loss(
            real_pred, True, is_disc=True) + self.gan_loss(
                fake_pred, False, is_disc=True)
        losses_dict['loss_disc'] = loss_disc.detach()
        losses_dict['real_score'] = real_pred.detach().mean()
        losses_dict['fake_score'] = fake_pred.detach().mean()
        loss_disc.backward()

        if (curr_iter + 1) % self.net_d_reg_every == 0:
            gt_imgs.requires_grad = True

            gen_accu_iters = gen_optimizer_wrapper._accumulative_counts
            gen_optimizer_wrapper.initialize_count_status(
                self.generator, 0, self.generator_steps * gen_accu_iters)

            real_pred = self.discriminator(gt_imgs)
            loss_disc_r1 = self.r1_penalty(real_pred, gt_imgs)
            loss_disc_r1 = (
                self.r1_reg_weight / 2 * loss_disc_r1 * self.net_d_reg_every +
                0 * real_pred[0])
            # NOTE: why do we need to add 0 * real_pred, otherwise, a runtime
            # error will arise: RuntimeError: Expected to have finished
            # reduction in the prior iteration before starting a new one.
            # This error indicates that your module has parameters that were
            # not used in producing loss.
            losses_dict['loss_disc_r1'] = loss_disc_r1.detach().mean()
            loss_disc_r1.backward()

        disc_optimizer_wrapper.step()
        set_requires_grad(self.discriminator, False)
        self.generator.zero_grad()

        fake_imgs, _ = self.generator(input_imgs, conditions)
        fake_pred = self.discriminator(fake_imgs)

        loss_gen = self.gan_loss(fake_pred, True, is_disc=False)
        losses_dict['loss_gen_wgan'] = loss_gen.detach()

        if self.pixel_loss:
            loss_gen_pixel = self.pixel_loss(fake_imgs, gt_imgs)
            loss_gen += loss_gen_pixel
            losses_dict['loss_gen_pixel'] = loss_gen_pixel.detach()

        if self.perceptual_loss:
            loss_gen_perceptual, loss_gen_style = self.perceptual_loss(
                fake_imgs, gt_imgs)
            if loss_gen_perceptual is not None:
                loss_gen += loss_gen_perceptual
                losses_dict[
                    'loss_gen_perceptual'] = loss_gen_perceptual.detach()
            if loss_gen_style is not None:
                loss_gen += loss_gen_style
                losses_dict['loss_gen_style'] = loss_gen_style.detach()

        loss_gen.backward()
        gen_optimizer_wrapper.step()

        # only do ema after generator update
        if self.with_ema_gen and (curr_iter + 1) >= (
                self.ema_start * self.discriminator_steps * disc_accu_iters):
            self.generator_ema.update_parameters(
                self.generator.
                module if is_model_wrapper(self.generator) else self.generator)
            # if not update buffer, copy buffer from orig model
            if not self.generator_ema.update_buffers:
                self.generator_ema.sync_buffers(
                    self.generator.module
                    if is_model_wrapper(self.generator) else self.generator)
        elif self.with_ema_gen:
            # before ema, copy weights from orig
            self.generator_ema.sync_parameters(
                self.generator.
                module if is_model_wrapper(self.generator) else self.generator)

        _, log_vars = self.parse_losses(losses_dict)

        return log_vars
