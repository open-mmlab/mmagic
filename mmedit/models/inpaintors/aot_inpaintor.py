# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from pathlib import Path

import mmcv
import torch

from mmedit.core import tensor2img
from ..common import set_requires_grad
from ..registry import MODELS
from .one_stage import OneStageInpaintor


@MODELS.register_module()
class AOTInpaintor(OneStageInpaintor):
    """Inpaintor for AOT-GAN method.

    This inpaintor is implemented according to the paper: Aggregated Contextual
    Transformations for High-Resolution Image Inpainting
    """

    def forward_train_d(self, data_batch, is_real, is_disc, mask):
        """Forward function in discriminator training step.

        In this function, we compute the prediction for each data batch (real
        or fake). Meanwhile, the standard gan loss will be computed with
        several proposed losses for stable training.

        Args:
            data (torch.Tensor): Batch of real data or fake data.
            is_real (bool): If True, the gan loss will regard this batch as
                real data. Otherwise, the gan loss will regard this batch as
                fake data.
            is_disc (bool): If True, this function is called in discriminator
                training step. Otherwise, this function is called in generator
                training step. This will help us to compute different types of
                adversarial loss, like LSGAN.
            mask (torch.Tensor): Mask of data.

        Returns:
            dict: Contains the loss items computed in this function.
        """

        pred = self.disc(data_batch)
        loss_ = self.loss_gan(pred, is_real, is_disc, mask=mask)

        loss = dict(real_loss=loss_) if is_real else dict(fake_loss=loss_)

        if self.with_disc_shift_loss:
            loss_d_shift = self.loss_disc_shift(loss_)
            # 0.5 for average the fake and real data
            loss.update(loss_disc_shift=loss_d_shift * 0.5)

        return loss

    def generator_loss(self, fake_res, fake_img, data_batch):
        """Forward function in generator training step.

        In this function, we mainly compute the loss items for generator with
        the given (fake_res, fake_img). In general, the `fake_res` is the
        direct output of the generator and the `fake_img` is the composition of
        direct output and ground-truth image.

        Args:
            fake_res (torch.Tensor): Direct output of the generator.
            fake_img (torch.Tensor): Composition of `fake_res` and
                ground-truth image.
            data_batch (dict): Contain other elements for computing losses.

        Returns:
            tuple(dict): Dict contains the results computed within this
                function for visualization and dict contains the loss items
                computed in this function.
        """
        gt = data_batch['gt_img']
        mask = data_batch['mask']
        masked_img = data_batch['masked_img']

        loss = dict()

        if self.with_gan:
            pred = self.disc(fake_img)
            loss_g_fake = self.loss_gan(pred, True, False, mask=mask)
            loss['loss_g_fake'] = loss_g_fake

        if self.with_l1_valid_loss:
            loss_l1_valid = self.loss_l1_valid(fake_res, gt)
            loss['loss_l1_valid'] = loss_l1_valid

        if self.with_out_percep_loss:
            loss_out_percep, loss_out_style = self.loss_percep(fake_res, gt)
            if loss_out_percep is not None:
                loss['loss_out_percep'] = loss_out_percep
            if loss_out_style is not None:
                loss['loss_out_style'] = loss_out_style

        res = dict(
            gt_img=gt.cpu(),
            masked_img=masked_img.cpu(),
            fake_res=fake_res.cpu(),
            fake_img=fake_img.cpu())

        return res, loss

    def forward_test(self,
                     masked_img,
                     mask,
                     save_image=False,
                     save_path=None,
                     iteration=None,
                     **kwargs):
        """Forward function for testing.

        Args:
            masked_img (torch.Tensor): Tensor with shape of (n, 3, h, w).
            mask (torch.Tensor): Tensor with shape of (n, 1, h, w).
            save_image (bool, optional): If True, results will be saved as
                image. Default: False.
            save_path (str, optional): If given a valid str, the reuslts will
                be saved in this path. Default: None.
            iteration (int, optional): Iteration number. Default: None.

        Returns:
            dict: Contain output results and eval metrics (if exist).
        """

        masked_img = masked_img.float() + mask
        input_x = torch.cat([masked_img, mask], dim=1)

        fake_res = self.generator(input_x)
        fake_img = fake_res * mask + masked_img * (1. - mask)

        output = dict()
        eval_results = {}
        if self.eval_with_metrics:
            gt_img = kwargs['gt_img']
            data_dict = dict(
                gt_img=gt_img, fake_res=fake_res, fake_img=fake_img, mask=None)
            for metric_name in self.test_cfg['metrics']:
                if metric_name in ['ssim', 'psnr']:
                    eval_results[metric_name] = self._eval_metrics[
                        metric_name](tensor2img(fake_img, min_max=(-1, 1)),
                                     tensor2img(gt_img, min_max=(-1, 1)))
                else:
                    eval_results[metric_name] = self._eval_metrics[
                        metric_name]()(data_dict).item()
            output['eval_result'] = eval_results
        else:
            output['fake_res'] = fake_res
            output['fake_img'] = fake_img

        output['meta'] = None if 'meta' not in kwargs else kwargs['meta'][0]

        if save_image:
            assert save_image and save_path is not None, (
                'Save path should been given')
            assert output['meta'] is not None, (
                'Meta information should be given to save image.')

            tmp_filename = output['meta']['gt_img_path']
            filestem = Path(tmp_filename).stem
            if iteration is not None:
                filename = f'{filestem}_{iteration}.png'
            else:
                filename = f'{filestem}.png'
            mmcv.mkdir_or_exist(save_path)
            img_list = [kwargs['gt_img']] if 'gt_img' in kwargs else []
            img_list.extend(
                [masked_img,
                 mask.expand_as(masked_img), fake_res, fake_img])
            img = torch.cat(img_list, dim=3).cpu()
            self.save_visualization(img, osp.join(save_path, filename))
            output['save_img_path'] = osp.abspath(
                osp.join(save_path, filename))

        return output

    def train_step(self, data_batch, optimizer):
        """Train step function.

        In this function, the inpaintor will finish the train step following
        the pipeline:
        1. get fake res/image
        2. compute reconstruction losses for generator
        3. compute adversarial loss for discriminator
        4. optimize generator
        5. optimize discriminator

        Args:
            data_batch (torch.Tensor): Batch of data as input.
            optimizer (dict[torch.optim.Optimizer]): Dict with optimizers for
                generator and discriminator (if exist).

        Returns:
            dict: Dict with loss, information for logger, the number of
                samples and results for visualization.
        """
        log_vars = {}

        gt_img = data_batch['gt_img']
        mask = data_batch['mask']
        masked_img = data_batch['masked_img']
        masked_img = masked_img.float() + mask

        # get common output from encdec
        input_x = torch.cat([masked_img, mask], dim=1)
        fake_res = self.generator(input_x)
        fake_img = gt_img * (1. - mask) + fake_res * mask

        # discriminator training step
        if self.train_cfg.disc_step > 0:
            set_requires_grad(self.disc, True)
            disc_losses_real = self.forward_train_d(
                gt_img, True, True, mask=mask)
            disc_losses_fake = self.forward_train_d(
                fake_img.detach(), False, True, mask=mask)
            disc_losses_ = disc_losses_real['real_loss'] + disc_losses_fake[
                'fake_loss']
            disc_losses = dict(disc_losses=disc_losses_)
            loss_disc, log_vars_d = self.parse_losses(disc_losses)
            log_vars.update(log_vars_d)
            optimizer['disc'].zero_grad()
            loss_disc.backward()

            optimizer['disc'].step()

            self.disc_step_count = (self.disc_step_count +
                                    1) % self.train_cfg.disc_step
            if self.disc_step_count != 0:
                # results contain the data for visualization
                results = dict(
                    gt_img=gt_img.cpu(),
                    masked_img=masked_img.cpu(),
                    fake_res=fake_res.cpu(),
                    fake_img=fake_img.cpu())
                outputs = dict(
                    log_vars=log_vars,
                    num_samples=len(data_batch['gt_img'].data),
                    results=results)

                return outputs

        # generator (encdec) training step, results contain the data
        # for visualization
        if self.with_gan:
            set_requires_grad(self.disc, False)
        results, g_losses = self.generator_loss(fake_res, fake_img, data_batch)
        loss_g, log_vars_g = self.parse_losses(g_losses)
        log_vars.update(log_vars_g)
        optimizer['generator'].zero_grad()
        loss_g.backward()
        optimizer['generator'].step()

        outputs = dict(
            log_vars=log_vars,
            num_samples=len(data_batch['gt_img'].data),
            results=results)

        return outputs
