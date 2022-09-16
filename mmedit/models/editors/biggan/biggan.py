# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine import Config
from mmengine.optim import OptimWrapper
from torch import Tensor

from mmedit.registry import MODELS
from mmedit.structures import EditDataSample
from ...base_models import BaseConditionalGAN

ModelType = Union[Dict, nn.Module]
TrainInput = Union[dict, Tensor]


@MODELS.register_module()
class BigGAN(BaseConditionalGAN):
    """Impelmentation of `Large Scale GAN Training for High Fidelity Natural
    Image Synthesis <https://arxiv.org/abs/1809.11096>`_ (BigGAN).

    Detailed architecture can be found in
    :class:~`mmgen.models.architectures.biggan.generator_discriminator.BigGANGenerator`  # noqa
    and
    :class:~`mmgen.models.architectures.biggan.generator_discriminator.BigGANDiscriminator`  # noqa

    Args:
        generator (ModelType): The config or model of the generator.
        discriminator (Optional[ModelType]): The config or model of the
            discriminator. Defaults to None.
        data_preprocessor (Optional[Union[dict, Config]]): The pre-process
            config or :class:`~mmgen.models.GANDataPreprocessor`.
        generator_steps (int): Number of times the generator was completely
            updated before the discriminator is updated. Defaults to 1.
        discriminator_steps (int): Number of times the discriminator was
            completely updated before the generator is updated. Defaults to 1.
        noise_size (Optional[int]): Size of the input noise vector.
            Default to 128.
        num_classes (Optional[int]): The number classes you would like to
            generate. Defaults to None.
        ema_config (Optional[Dict]): The config for generator's exponential
            moving average setting. Defaults to None.
    """

    def __init__(self,
                 generator: ModelType,
                 discriminator: Optional[ModelType] = None,
                 data_preprocessor: Optional[Union[dict, Config]] = None,
                 generator_steps: int = 1,
                 discriminator_steps: int = 1,
                 noise_size: Optional[int] = None,
                 num_classes: Optional[int] = None,
                 ema_config: Optional[Dict] = None):
        super().__init__(generator, discriminator, data_preprocessor,
                         generator_steps, discriminator_steps, noise_size,
                         num_classes, ema_config)

    def disc_loss(self, disc_pred_fake: Tensor,
                  disc_pred_real: Tensor) -> Tuple:
        r"""Get disc loss. BigGAN use hinge loss to train
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
        r"""Get disc loss. BigGAN use hinge loss to train
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
        real_labels = self.data_sample_to_label(data_samples)
        assert real_labels is not None, (
            'Cannot found \'gt_label\' in \'data_sample\'.')

        num_batches = real_imgs.shape[0]

        noise_batch = self.noise_fn(num_batches=num_batches)
        fake_labels = self.label_fn(num_batches=num_batches)
        with torch.no_grad():
            fake_imgs = self.generator(
                noise=noise_batch, label=fake_labels, return_noise=False)

        disc_pred_fake = self.discriminator(fake_imgs, label=fake_labels)
        disc_pred_real = self.discriminator(real_imgs, label=real_labels)

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
        fake_labels = self.label_fn(num_batches=num_batches)
        fake_imgs = self.generator(
            noise=noise, label=fake_labels, return_noise=False)

        disc_pred_fake = self.discriminator(fake_imgs, label=fake_labels)
        parsed_loss, log_vars = self.gen_loss(disc_pred_fake)

        optimizer_wrapper.update_params(parsed_loss)
        return log_vars
