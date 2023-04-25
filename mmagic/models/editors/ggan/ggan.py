# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from mmengine.optim import OptimWrapper
from torch import Tensor

from mmagic.registry import MODELS
from mmagic.structures import DataSample
from ...base_models import BaseGAN


@MODELS.register_module()
class GGAN(BaseGAN):
    """Implementation of `Geometric GAN`.

    <https://arxiv.org/abs/1705.02894>`_(GGAN).
    """

    def disc_loss(self, disc_pred_fake: Tensor,
                  disc_pred_real: Tensor) -> Tuple:
        r"""Get disc loss. GGAN use hinge loss to train
        the discriminator.

        .. math:
            L_{D} = -\mathbb{E}_{\left(x, y\right)\sim{p}_{data}}
                \left[\min\left(0, -1 + D\left(x, y\right)\right)\right]
                -\mathbb{E}_{z\sim{p_{z}}, y\sim{p_{data}}}\left[\min
                \left(0, -1 - D\left(G\left(z\right), y\right)\right)\right]

        Args:
            disc_pred_fake (Tensor): Discriminator's prediction of the fake
                images.
            disc_pred_real (Tensor): Discriminator's prediction of the real
                images.

        Returns:
            tuple[Tensor, dict]: Loss value and a dict of log variables.
        """
        losses_dict = dict()
        losses_dict['loss_disc_fake'] = F.relu(1 + disc_pred_fake).mean()
        losses_dict['loss_disc_real'] = F.relu(1 - disc_pred_real).mean()

        loss, log_var = self.parse_losses(losses_dict)
        return loss, log_var

    def gen_loss(self, disc_pred_fake):
        r"""Get disc loss. GGAN use hinge loss to train
        the generator.

        .. math:
            L_{G} = -\mathbb{E}_{z\sim{p_{z}}, y\sim{p_{data}}}
                D\left(G\left(z\right), y\right)

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

    def train_discriminator(self, inputs: dict, data_samples: List[DataSample],
                            optimizer_wrapper: OptimWrapper
                            ) -> Dict[str, Tensor]:
        """Train discriminator.

        Args:
            inputs (dict): Inputs from dataloader.
            data_samples (List[DataSample]): Data samples from dataloader.
            optim_wrapper (OptimWrapper): OptimWrapper instance used to update
                model parameters.
        Returns:
            Dict[str, Tensor]: A ``dict`` of tensor for logging.
        """
        real_imgs = data_samples.gt_img

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

    def train_generator(self, inputs: dict, data_samples: List[DataSample],
                        optimizer_wrapper: OptimWrapper) -> Dict[str, Tensor]:
        """Train generator.

        Args:
            inputs (dict): Inputs from dataloader.
            data_samples (List[DataSample]): Data samples from dataloader.
                Do not used in generator's training.
            optim_wrapper (OptimWrapper): OptimWrapper instance used to update
                model parameters.

        Returns:
            Dict[str, Tensor]: A ``dict`` of tensor for logging.
        """
        num_batches = len(data_samples)

        noise = self.noise_fn(num_batches=num_batches)
        fake_imgs = self.generator(noise=noise, return_noise=False)

        disc_pred_fake = self.discriminator(fake_imgs)
        parsed_loss, log_vars = self.gen_loss(disc_pred_fake)

        optimizer_wrapper.update_params(parsed_loss)
        return log_vars
