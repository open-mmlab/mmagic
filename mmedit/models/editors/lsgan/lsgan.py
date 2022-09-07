# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from mmengine.optim import OptimWrapper
from torch import Tensor

from mmedit.registry import MODELS
from mmedit.structures import EditDataSample
from ...base_models import BaseGAN


@MODELS.register_module()
class LSGAN(BaseGAN):
    """Impelmentation of `Least Squares Generative Adversarial Networks`.

    Paper link: https://arxiv.org/pdf/1611.04076.pdf

    Detailed architecture can be found in
    :class:~`mmgen.models.architectures.lsgan.generator_generator.LSGANGenerator`  # noqa
    and
    :class:~`mmgen.models.architectures.lsgan.generator_discriminator.LSGANDiscriminator`  # noqa
    """

    def disc_loss(self, disc_pred_fake: Tensor,
                  disc_pred_real: Tensor) -> Tuple:
        r"""Get disc loss. LSGAN use the least squares loss to train
        the discriminator.

        .. math::
            L_{D}=\left(D\left(X_{\text {data }}\right)-1\right)^{2}
                +(D(G(z)))^{2}


        Args:
            disc_pred_fake (Tensor): Discriminator's prediction of the fake
                images.
            disc_pred_real (Tensor): Discriminator's prediction of the real
                images.

        Returns:
            tuple[Tensor, dict]: Loss value and a dict of log variables.
        """
        losses_dict = dict()
        losses_dict['loss_disc_fake'] = F.mse_loss(
            disc_pred_fake, 0. * torch.ones_like(disc_pred_fake))
        losses_dict['loss_disc_real'] = F.mse_loss(
            disc_pred_real, 1. * torch.ones_like(disc_pred_real))

        loss, log_var = self.parse_losses(losses_dict)
        return loss, log_var

    def gen_loss(self, disc_pred_fake: Tensor) -> Tuple:
        """Get gen loss. LSGAN use the least squares loss to train the
        generator.

        .. math::
            L_{G}=(D(G(z))-1)^{2}

        Args:
            disc_pred_fake (Tensor): Discriminator's prediction of the fake
                images.

        Returns:
            tuple[Tensor, dict]: Loss value and a dict of log variables.
        """
        losses_dict = dict()
        losses_dict['loss_gen'] = F.mse_loss(
            disc_pred_fake, 1. * torch.ones_like(disc_pred_fake))
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

        parsed_losses, log_vars = self.disc_loss(disc_pred_fake,
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
