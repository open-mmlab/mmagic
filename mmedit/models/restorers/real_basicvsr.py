# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn.functional as F
from mmcv.parallel import is_module_wrapper

from ..builder import build_loss
from ..common import set_requires_grad
from ..registry import MODELS
from .real_esrgan import RealESRGAN


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
        pretrained (str): Path for pretrained model. Default: None.
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
                 is_use_ema=True,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):

        super().__init__(generator, discriminator, gan_loss, pixel_loss,
                         perceptual_loss, is_use_sharpened_gt_in_pixel,
                         is_use_sharpened_gt_in_percep,
                         is_use_sharpened_gt_in_gan, is_use_ema, train_cfg,
                         test_cfg, pretrained)

        self.cleaning_loss = build_loss(
            cleaning_loss) if cleaning_loss else None

    def train_step(self, data_batch, optimizer):
        """Train step.

        Args:
            data_batch (dict): A batch of data.
            optimizer (obj): Optimizer.

        Returns:
            dict: Returned output.
        """

        # during initialization, load weights from the ema model
        if (self.step_counter == self.start_iter
                and self.generator_ema is not None):
            if is_module_wrapper(self.generator):
                self.generator.module.load_state_dict(
                    self.generator_ema.module.state_dict())
            else:
                self.generator.load_state_dict(self.generator_ema.state_dict())

        # data
        lq = data_batch['lq']
        gt = data_batch['gt']

        gt_pixel, gt_percep, gt_gan = gt.clone(), gt.clone(), gt.clone()
        if self.is_use_sharpened_gt_in_pixel:
            gt_pixel = data_batch['gt_unsharp']
        if self.is_use_sharpened_gt_in_percep:
            gt_percep = data_batch['gt_unsharp']
        if self.is_use_sharpened_gt_in_gan:
            gt_gan = data_batch['gt_unsharp']

        if self.cleaning_loss:
            n, t, c, h, w = gt.size()
            gt_clean = gt_pixel.view(-1, c, h, w)
            gt_clean = F.interpolate(gt_clean, scale_factor=0.25, mode='area')
            gt_clean = gt_clean.view(n, t, c, h // 4, w // 4)

        # generator
        fake_g_output, fake_g_lq = self.generator(lq, return_lqs=True)
        losses = dict()
        log_vars = dict()

        # reshape: (n, t, c, h, w) -> (n*t, c, h, w)
        c, h, w = gt.shape[2:]
        gt_pixel = gt_pixel.view(-1, c, h, w)
        gt_percep = gt_percep.view(-1, c, h, w)
        gt_gan = gt_gan.view(-1, c, h, w)
        fake_g_output = fake_g_output.view(-1, c, h, w)

        # no updates to discriminator parameters
        if self.gan_loss:
            set_requires_grad(self.discriminator, False)

        if (self.step_counter % self.disc_steps == 0
                and self.step_counter >= self.disc_init_steps):
            if self.pixel_loss:
                losses['loss_pix'] = self.pixel_loss(fake_g_output, gt_pixel)
            if self.cleaning_loss:
                losses['loss_clean'] = self.cleaning_loss(fake_g_lq, gt_clean)
            if self.perceptual_loss:
                loss_percep, loss_style = self.perceptual_loss(
                    fake_g_output, gt_percep)
                if loss_percep is not None:
                    losses['loss_perceptual'] = loss_percep
                if loss_style is not None:
                    losses['loss_style'] = loss_style

            # gan loss for generator
            if self.gan_loss:
                fake_g_pred = self.discriminator(fake_g_output)
                losses['loss_gan'] = self.gan_loss(
                    fake_g_pred, target_is_real=True, is_disc=False)

            # parse loss
            loss_g, log_vars_g = self.parse_losses(losses)
            log_vars.update(log_vars_g)

            # optimize
            optimizer['generator'].zero_grad()
            loss_g.backward()
            optimizer['generator'].step()

        # discriminator
        if self.gan_loss:
            set_requires_grad(self.discriminator, True)
            # real
            real_d_pred = self.discriminator(gt_gan)
            loss_d_real = self.gan_loss(
                real_d_pred, target_is_real=True, is_disc=True)
            loss_d, log_vars_d = self.parse_losses(
                dict(loss_d_real=loss_d_real))
            optimizer['discriminator'].zero_grad()
            loss_d.backward()
            log_vars.update(log_vars_d)

            # fake
            fake_d_pred = self.discriminator(fake_g_output.detach())
            loss_d_fake = self.gan_loss(
                fake_d_pred, target_is_real=False, is_disc=True)
            loss_d, log_vars_d = self.parse_losses(
                dict(loss_d_fake=loss_d_fake))
            loss_d.backward()
            log_vars.update(log_vars_d)

            optimizer['discriminator'].step()

        self.step_counter += 1

        log_vars.pop('loss')  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars,
            num_samples=len(gt.data),
            results=dict(lq=lq.cpu(), gt=gt.cpu(), output=fake_g_output.cpu()))

        return outputs
