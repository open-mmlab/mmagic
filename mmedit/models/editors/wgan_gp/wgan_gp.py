# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

import torch
from mmengine.optim import OptimWrapper
from torch import Tensor

from mmedit.models.base_models import BaseGAN
from mmedit.models.losses import gradient_penalty_loss
from mmedit.registry import MODELS
from mmedit.structures import EditDataSample


@MODELS.register_module()
class WGANGP(BaseGAN):
    """Impelmentation of `Improved Training of Wasserstein GANs`.

    Paper link: https://arxiv.org/pdf/1704.00028

    Detailed architecture can be found in
    :class:~`mmgen.models.architectures.wgan_gp.generator_discriminator.WGANGPGenerator`  # noqa
    and
    :class:~`mmgen.models.architectures.wgan_gp.generator_discriminator.WGANGPDiscriminator`  # noqa
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # gradient penalty loss settings
        self.gp_norm_mode = self.loss_config.get('norm_mode', 'HWC')
        self.gp_loss_weight = self.loss_config.get('loss_weight', 10)

    def disc_loss(self, real_data: Tensor, fake_data: Tensor,
                  disc_pred_fake: Tensor, disc_pred_real: Tensor) -> Tuple:
        r"""Get disc loss. WGAN-GP use the wgan loss and gradient penalty to
        train the discriminator.

        Args:
            real_data (Tensor): Real input data.
            fake_data (Tensor): Fake input data.
            disc_pred_fake (Tensor): Discriminator's prediction of the fake
                images.
            disc_pred_real (Tensor): Discriminator's prediction of the real
                images.

        Returns:
            tuple[Tensor, dict]: Loss value and a dict of log variables.
        """
        losses_dict = dict()
        losses_dict['loss_disc_fake'] = disc_pred_fake.mean()
        losses_dict['loss_disc_real'] = -disc_pred_real.mean()

        # Gradient Penalty loss
        losses_dict['loss_gp'] = self.gp_loss_weight * gradient_penalty_loss(
            self.discriminator,
            real_data,
            fake_data,
            norm_mode=self.gp_norm_mode)
        loss, log_var = self.parse_losses(losses_dict)
        return loss, log_var

    def gen_loss(self, disc_pred_fake: Tensor) -> Tuple:
        """Get gen loss. DCGAN use the wgan loss to train the generator.

        Args:
            disc_pred_fake (Tensor): Discriminator's prediction of the fake
                images.

        Returns:
            tuple[Tensor, dict]: Loss value and a dict of log variables.
        """
        losses_dict = dict()
        losses_dict['loss_gen'] = -disc_pred_fake.mean()
        loss, log_var = self.parse_losses(losses_dict)
        return loss, log_var

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
            fake_imgs = self.generator(noise=noise_batch, return_noise=False)

        disc_pred_fake = self.discriminator(fake_imgs)
        disc_pred_real = self.discriminator(real_imgs)

        parsed_losses, log_vars = self.disc_loss(real_imgs, fake_imgs,
                                                 disc_pred_fake,
                                                 disc_pred_real)
        optimizer_wrapper.update_params(parsed_losses)
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
        fake_imgs = self.generator(noise=noise, return_noise=False)

        disc_pred_fake = self.discriminator(fake_imgs)
        parsed_loss, log_vars = self.gen_loss(disc_pred_fake)

        optimizer_wrapper.update_params(parsed_loss)
        return log_vars
