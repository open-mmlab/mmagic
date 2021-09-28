import numbers
import os.path as osp

import mmcv
import torch

from mmedit.core import tensor2img
from ..common import set_requires_grad
from ..registry import MODELS
from .srgan import SRGAN


@MODELS.register_module()
class DFD(SRGAN):
    """DFD model for Face Super-Resolution.

    Paper: Blind Face Restoration via Deep Multi-scale Component Dictionaries.

    Args:
        generator (dict): Config for the generator.
        pixel_loss (dict): Config for the pixel loss.
        discriminator (dict): Config for the discriminator. Default: None.
        gan_loss (dict): Config for the gan loss. Default: None.
        perceptual_loss (dict): Config for the perceptual loss. Default: None.
        train_cfg (dict): Config for train. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path for pretrained model. Default: None.
    """

    def __init__(self,
                 generator,
                 pixel_loss,
                 discriminator=None,
                 gan_loss=None,
                 perceptual_loss=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__(
            generator=generator,
            discriminator=discriminator,
            gan_loss=gan_loss,
            pixel_loss=pixel_loss,
            perceptual_loss=perceptual_loss,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
        self.fix_iter = train_cfg.get('fix_iter', 0) if train_cfg else 0

    def forward(self, lq, gt=None, test_mode=False, location=None, **kwargs):
        """Forward function.

        Args:
            lq (Tensor): Input lq images.
            gt (Tensor): Ground-truth image. Default: None.
            test_mode (bool): Whether in test mode or not. Default: False.
            kwargs (dict): Other arguments.
        """

        if test_mode:
            return self.forward_test(lq, gt=gt, location=location, **kwargs)

        return self.generator.forward(lq, location)

    def forward_test(self,
                     lq,
                     gt=None,
                     location=None,
                     meta=None,
                     save_image=False,
                     save_path=None,
                     iteration=None):
        """Testing forward function.

        Args:
            lq (Tensor): LQ image.
            gt (Tensor): GT image.
            meta (list[dict]): Meta data, such as path of GT file.
                Default: None.
            save_image (bool): Whether to save image. Default: False.
            save_path (str): Path to save image. Default: None.
            iteration (int): Iteration for the saving image name.
                Default: None.

        Returns:
            dict: Output results, which contain either key(s)
                1. 'eval_result'.
                2. 'lq', 'pred'.
                3. 'lq', 'pred', 'gt'.
        """

        # generator
        with torch.no_grad():
            pred = self.generator.forward(lq, location)
            pred = pred / 2. + 0.5
            if gt is not None:
                gt = gt / 2. + 0.5
        if self.test_cfg is not None and self.test_cfg.get('metrics', None):
            assert gt is not None, (
                'evaluation with metrics must have gt images.')
            results = dict(eval_result=self.evaluate(pred, gt))
        else:
            results = dict(lq=lq.cpu(), output=pred.cpu())
            if gt is not None:
                results['gt'] = gt.cpu()

        # save image
        if save_image:
            if 'gt_path' in meta[0]:
                pred_path = meta[0]['gt_path']
            else:
                pred_path = meta[0]['lq_path']
            folder_name = osp.splitext(osp.basename(pred_path))[0]
            if isinstance(iteration, numbers.Number):
                save_path = osp.join(save_path, folder_name,
                                     f'{folder_name}-{iteration + 1:06d}.png')
            elif iteration is None:
                save_path = osp.join(save_path, f'{folder_name}.png')
            else:
                raise ValueError('iteration should be number or None, '
                                 f'but got {type(iteration)}')
            mmcv.imwrite(tensor2img(pred), save_path)
        return results

    def train_step(self, data_batch, optimizer):
        """Train step.

        Args:
            data_batch (dict): A batch of data, which requires
                'lq', 'gt'
            optimizer (obj): Optimizer.

        Returns:
            dict: Returned output, which includes:
                log_vars, num_samples, results (lq, gt and pred).

        """
        # data
        lq = data_batch['lq']
        gt = data_batch['gt']

        # generate
        pred = self(**data_batch, test_mode=False)

        # loss
        losses = dict()
        log_vars = dict()
        losses['loss_pixel'] = self.pixel_loss(pred, gt)

        if self.step_counter >= self.fix_iter:
            # perceptual loss
            if self.perceptual_loss:
                loss_perceptual = self.perceptual_loss(pred, gt)[0]
                losses['loss_perceptual'] = loss_perceptual
            # gan loss for generator
            if self.gan_loss:
                fake_g_pred = self.discriminator(pred)
                losses['loss_gan'] = self.gan_loss(
                    fake_g_pred, target_is_real=True, is_disc=False)

        # parse loss
        loss_g, log_vars_g = self.parse_losses(losses)
        log_vars.update(log_vars_g)

        # optimize
        optimizer['generator'].zero_grad()
        loss_g.backward()
        optimizer['generator'].step()

        if self.discriminator and self.step_counter >= self.fix_iter:
            # discriminator
            set_requires_grad(self.discriminator, True)
            for _ in range(self.disc_steps):
                # real
                real_d_pred = self.discriminator(gt)
                loss_d_real = self.gan_loss(
                    real_d_pred, target_is_real=True, is_disc=True)
                loss_d, log_vars_d = self.parse_losses(
                    dict(loss_d_real=loss_d_real))
                optimizer['discriminator'].zero_grad()
                loss_d.backward()
                log_vars.update(log_vars_d)
                # fake
                fake_d_pred = self.discriminator(pred.detach())
                loss_d_fake = self.gan_loss(
                    fake_d_pred, target_is_real=False, is_disc=True)
                loss_d, log_vars_d = self.parse_losses(
                    dict(loss_d_fake=loss_d_fake))
                loss_d.backward()
                log_vars.update(log_vars_d)

                optimizer['discriminator'].step()

        log_vars.pop('loss')  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars,
            num_samples=len(gt.data),
            results=dict(lq=lq.cpu(), gt=gt.cpu(), output=pred.cpu()))

        self.step_counter += 1

        return outputs
