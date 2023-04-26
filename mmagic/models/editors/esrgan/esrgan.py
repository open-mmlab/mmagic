# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmagic.registry import MODELS
from ..srgan import SRGAN


@MODELS.register_module()
class ESRGAN(SRGAN):
    """Enhanced SRGAN model for single image super-resolution.

    Ref:
    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.
    It uses RaGAN for GAN updates:
    The relativistic discriminator: a key element missing from standard GAN.

    Args:
        generator (dict): Config for the generator.
        discriminator (dict): Config for the discriminator. Default: None.
        gan_loss (dict): Config for the gan loss.
            Note that the loss weight in gan loss is only for the generator.
        pixel_loss (dict): Config for the pixel loss. Default: None.
        perceptual_loss (dict): Config for the perceptual loss. Default: None.
        train_cfg (dict): Config for training. Default: None.
            You may change the training of gan by setting:
            `disc_steps`: how many discriminator updates after one generate
            update;
            `disc_init_steps`: how many discriminator updates at the start of
            the training.
            These two keys are useful when training with WGAN.
        test_cfg (dict): Config for testing. Default: None.
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`. Default: None.
    """

    def g_step(self, batch_outputs: torch.Tensor, batch_gt_data: torch.Tensor):
        """G step of GAN: Calculate losses of generator.

        Args:
            batch_outputs (Tensor): Batch output of generator.
            batch_gt_data (Tensor): Batch GT data.

        Returns:
            dict: Dict of losses.
        """

        losses = dict()

        # pix loss
        if self.pixel_loss:
            losses['loss_pix'] = self.pixel_loss(batch_outputs, batch_gt_data)

        # perceptual loss
        if self.perceptual_loss:
            loss_percep, loss_style = self.perceptual_loss(
                batch_outputs, batch_gt_data)
            if loss_percep is not None:
                losses['loss_perceptual'] = loss_percep
            if loss_style is not None:
                losses['loss_style'] = loss_style

        # gan loss for generator
        if self.gan_loss and self.discriminator:
            real_d_pred = self.discriminator(batch_gt_data).detach()
            fake_g_pred = self.discriminator(batch_outputs)
            loss_gan_fake = self.gan_loss(
                fake_g_pred - torch.mean(real_d_pred),
                target_is_real=True,
                is_disc=False)
            loss_gan_real = self.gan_loss(
                real_d_pred - torch.mean(fake_g_pred),
                target_is_real=False,
                is_disc=False)
            losses['loss_gan'] = (loss_gan_fake + loss_gan_real) / 2

        return losses

    def d_step_real(self, batch_outputs: torch.Tensor,
                    batch_gt_data: torch.Tensor):
        """D step of real data.

        Args:
            batch_outputs (Tensor): Batch output of generator.
            batch_gt_data (Tensor): Batch GT data.

        Returns:
            dict: Dict of losses.
        """

        # real
        fake_d_pred = self.discriminator(batch_outputs)
        real_d_pred = self.discriminator(batch_gt_data)
        loss_d_real = self.gan_loss(
            real_d_pred - torch.mean(fake_d_pred.detach()),
            target_is_real=True,
            is_disc=True
        ) * 0.5  # 0.5 for averaging loss_d_real and loss_d_fake

        self.real_d_pred = torch.mean(real_d_pred.detach())
        # for d_step_fake

        return loss_d_real

    def d_step_fake(self, batch_outputs: torch.Tensor, batch_gt_data):
        """D step of fake data.

        Args:
            batch_outputs (Tensor): Batch output of generator.
            batch_gt_data (Tensor): Batch GT data.

        Returns:
            dict: Dict of losses.
        """

        # fake
        fake_d_pred = self.discriminator(batch_outputs.detach())
        loss_d_fake = self.gan_loss(
            fake_d_pred - self.real_d_pred, target_is_real=False, is_disc=True
        ) * 0.5  # 0.5 for averaging loss_d_real and loss_d_fake

        return loss_d_fake
