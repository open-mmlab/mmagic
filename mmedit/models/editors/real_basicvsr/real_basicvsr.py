# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from mmengine.optim import OptimWrapperDict

from mmedit.registry import MODELS
from ..real_esrgan import RealESRGAN


@MODELS.register_module()
class RealBasicVSR(RealESRGAN):
    """RealBasicVSR model for real-world video super-resolution.

    Ref:
    Investigating Tradeoffs in Real-World Video Super-Resolution, arXiv

    Args:
        generator (dict): Config for the generator.
        discriminator (dict, optional): Config for the discriminator.
            Default: None.
        gan_loss (dict, optional): Config for the gan loss.
            Note that the loss weight in gan loss is only for the generator.
        pixel_loss (dict, optional): Config for the pixel loss. Default: None.
        cleaning_loss (dict, optional): Config for the image cleaning loss.
            Default: None.
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
                 cleaning_loss=None,
                 perceptual_loss=None,
                 is_use_sharpened_gt_in_pixel=False,
                 is_use_sharpened_gt_in_percep=False,
                 is_use_sharpened_gt_in_gan=False,
                 is_use_ema=False,
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
            is_use_sharpened_gt_in_pixel=is_use_sharpened_gt_in_pixel,
            is_use_sharpened_gt_in_percep=is_use_sharpened_gt_in_percep,
            is_use_sharpened_gt_in_gan=is_use_sharpened_gt_in_gan,
            is_use_ema=is_use_ema,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)

        self.cleaning_loss = MODELS.build(
            cleaning_loss) if cleaning_loss else None

    def extract_gt_data(self, data_samples):
        """extract gt data from data samples.

        Args:
            data_samples (list): List of EditDataSample.

        Returns:
            Tensor: Extract gt data.
        """

        gt_pixel, gt_percep, gt_gan = super().extract_gt_data(data_samples)
        n, t, c, h, w = gt_pixel.size()
        gt_pixel = gt_pixel.view(-1, c, h, w)
        gt_percep = gt_percep.view(-1, c, h, w)
        gt_gan = gt_gan.view(-1, c, h, w)

        if self.cleaning_loss:
            gt_clean = gt_pixel.view(-1, c, h, w)
            gt_clean = F.interpolate(
                gt_clean,
                scale_factor=0.25,
                mode='area',
                recompute_scale_factor=False)
            gt_clean = gt_clean.view(n, t, c, h // 4, w // 4)
        else:
            gt_clean = None

        return gt_pixel, gt_percep, gt_gan, gt_clean

    def g_step(self, batch_outputs, batch_gt_data):
        """G step of GAN: Calculate losses of generator.

        Args:
            batch_outputs (Tensor): Batch output of generator.
            batch_gt_data (Tuple[Tensor]): Batch GT data.

        Returns:
            dict: Dict of losses.
        """

        gt_pixel, gt_percep, gt_gan, gt_clean = batch_gt_data
        fake_g_output, fake_g_lq = batch_outputs
        fake_g_output = fake_g_output.view(gt_pixel.shape)

        losses = super().g_step(
            batch_outputs=fake_g_output,
            batch_gt_data=(gt_pixel, gt_percep, gt_gan))

        if self.cleaning_loss:
            losses['loss_clean'] = self.cleaning_loss(fake_g_lq, gt_clean)

        return losses

    def d_step_with_optim(self, batch_outputs: torch.Tensor,
                          batch_gt_data: torch.Tensor,
                          optim_wrapper: OptimWrapperDict):
        """D step with optim of GAN: Calculate losses of discriminator and run
        optim.

        Args:
            batch_outputs (Tensor): Batch output of generator.
            batch_gt_data (Tensor): Batch GT data.
            optim_wrapper (OptimWrapperDict): Optim wrapper dict.

        Returns:
            dict: Dict of parsed losses.
        """

        gt_pixel, gt_percep, gt_gan, gt_clean = batch_gt_data
        fake_g_output, fake_g_lq = batch_outputs
        fake_g_output = fake_g_output.view(gt_pixel.shape)

        return super().d_step_with_optim(
            batch_outputs=fake_g_output,
            batch_gt_data=(gt_pixel, gt_percep, gt_gan),
            optim_wrapper=optim_wrapper)

    def forward_train(self, batch_inputs, data_samples=None):
        """Forward Train.

        Run forward of generator with ``return_lqs=True``

        Args:
            batch_inputs (Tensor): Batch inputs.
            data_samples (List[EditDataSample]): Data samples of Editing.
                Default:None

        Returns:
            Tuple[Tensor]: Result of generator.
                (outputs, lqs)
        """

        return self.generator(batch_inputs, return_lqs=True)
