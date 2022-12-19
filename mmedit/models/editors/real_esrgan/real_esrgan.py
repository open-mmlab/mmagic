# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmedit.registry import MODELS
from ..srgan import SRGAN


@MODELS.register_module()
class RealESRGAN(SRGAN):
    """Real-ESRGAN model for single image super-resolution.

    Ref:
    Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure
    Synthetic Data, 2021.

    Note: generator_ema is realized in EMA_HOOK

    Args:
        generator (dict): Config for the generator.
        discriminator (dict, optional): Config for the discriminator.
            Default: None.
        gan_loss (dict, optional): Config for the gan loss.
            Note that the loss weight in gan loss is only for the generator.
        pixel_loss (dict, optional): Config for the pixel loss. Default: None.
        perceptual_loss (dict, optional): Config for the perceptual loss.
            Default: None.
        is_use_sharpened_gt_in_pixel (bool, optional): Whether to use the image
            sharpened by unsharp masking as the GT for pixel loss.
            Default: False.
        is_use_sharpened_gt_in_percep (bool, optional): Whether to use the
            image sharpened by unsharp masking as the GT for perceptual loss.
            Default: False.
        is_use_sharpened_gt_in_gan (bool, optional): Whether to use the
            image sharpened by unsharp masking as the GT for adversarial loss.
            Default: False.
        is_use_ema (bool, optional): When to apply exponential moving average
            on the network weights. Default: True.
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
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`. Default: None.
    """

    def __init__(self,
                 generator,
                 discriminator=None,
                 gan_loss=None,
                 pixel_loss=None,
                 perceptual_loss=None,
                 is_use_sharpened_gt_in_pixel=False,
                 is_use_sharpened_gt_in_percep=False,
                 is_use_sharpened_gt_in_gan=False,
                 is_use_ema=True,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 data_preprocessor=None):

        super().__init__(
            generator=generator,
            discriminator=discriminator,
            gan_loss=gan_loss,
            pixel_loss=pixel_loss,
            perceptual_loss=perceptual_loss,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)

        self.is_use_sharpened_gt_in_pixel = is_use_sharpened_gt_in_pixel
        self.is_use_sharpened_gt_in_percep = is_use_sharpened_gt_in_percep
        self.is_use_sharpened_gt_in_gan = is_use_sharpened_gt_in_gan
        self.is_use_ema = is_use_ema

        if train_cfg is not None:  # used for initializing from ema model
            self.start_iter = train_cfg.get('start_iter', -1)
        else:
            self.start_iter = -1

    def forward_tensor(self, inputs, data_samples=None, training=False):
        """Forward tensor. Returns result of simple forward.

        Args:
            inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (List[BaseDataElement], optional):
                data samples collated by :attr:`data_preprocessor`.
            training (bool): Whether is training. Default: False.

        Returns:
            Tensor: result of simple forward.
        """

        if training or not self.is_use_ema:
            feats = self.generator(inputs)
        else:
            feats = self.generator_ema(inputs)

        return feats

    def g_step(self, batch_outputs, batch_gt_data):
        """G step of GAN: Calculate losses of generator.

        Args:
            batch_outputs (Tensor): Batch output of generator.
            batch_gt_data (Tuple[Tensor]): Batch GT data.

        Returns:
            dict: Dict of losses.
        """

        gt_pixel, gt_percep, _ = batch_gt_data
        losses = dict()

        # pix loss
        if self.pixel_loss:
            losses['loss_pix'] = self.pixel_loss(batch_outputs, gt_pixel)

        # perceptual loss
        if self.perceptual_loss:
            loss_percep, loss_style = self.perceptual_loss(
                batch_outputs, gt_percep)
            if loss_percep is not None:
                losses['loss_perceptual'] = loss_percep
            if loss_style is not None:
                losses['loss_style'] = loss_style

        # gan loss for generator
        if self.gan_loss and self.discriminator:
            fake_g_pred = self.discriminator(batch_outputs)
            losses['loss_gan'] = self.gan_loss(
                fake_g_pred, target_is_real=True, is_disc=False)

        return losses

    def d_step_real(self, batch_outputs, batch_gt_data: torch.Tensor):
        """Real part of D step.

        Args:
            batch_outputs (Tensor): Batch output of generator.
            batch_gt_data (Tuple[Tensor]): Batch GT data.

        Returns:
            Tensor: Real part of gan_loss for discriminator.
        """

        _, _, gt_gan = batch_gt_data

        # real
        real_d_pred = self.discriminator(gt_gan)
        loss_d_real = self.gan_loss(
            real_d_pred, target_is_real=True, is_disc=True)

        return loss_d_real

    def d_step_fake(self, batch_outputs, batch_gt_data):
        """Fake part of D step.

        Args:
            batch_outputs (Tensor): Output of generator.
            batch_gt_data (Tuple[Tensor]): Batch GT data.

        Returns:
            Tensor: Fake part of gan_loss for discriminator.
        """

        # fake
        fake_d_pred = self.discriminator(batch_outputs.detach())
        loss_d_fake = self.gan_loss(
            fake_d_pred, target_is_real=False, is_disc=True)

        return loss_d_fake

    def extract_gt_data(self, data_samples):
        """extract gt data from data samples.

        Args:
            data_samples (list): List of EditDataSample.

        Returns:
            Tensor: Extract gt data.
        """

        gt_imgs = [data_sample.gt_img.data for data_sample in data_samples]
        gt = torch.stack(gt_imgs)
        gt_unsharp = [
            data_sample.gt_unsharp.data for data_sample in data_samples
        ]
        gt_unsharp = torch.stack(gt_unsharp)

        gt_pixel, gt_percep, gt_gan = gt.clone(), gt.clone(), gt.clone()
        if self.is_use_sharpened_gt_in_pixel:
            gt_pixel = gt_unsharp
        if self.is_use_sharpened_gt_in_percep:
            gt_percep = gt_unsharp
        if self.is_use_sharpened_gt_in_gan:
            gt_gan = gt_unsharp

        return gt_pixel, gt_percep, gt_gan
