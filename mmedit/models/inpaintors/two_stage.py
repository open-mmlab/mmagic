# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from pathlib import Path

import mmcv
import torch
from torchvision.utils import save_image

from mmedit.core import tensor2img
from ..common.model_utils import set_requires_grad
from ..registry import MODELS
from .one_stage import OneStageInpaintor


@MODELS.register_module()
class TwoStageInpaintor(OneStageInpaintor):
    """Two-Stage Inpaintor.

    Currently, we support these loss types in each of two stage inpaintors:
    ['loss_gan', 'loss_l1_hole', 'loss_l1_valid', 'loss_composed_percep',\
     'loss_out_percep', 'loss_tv']
    The `stage1_loss_type` and `stage2_loss_type` should be chosen from these
    loss types.

    Args:
        stage1_loss_type (tuple[str]): Contains the loss names used in the
            first stage model.
        stage2_loss_type (tuple[str]): Contains the loss names used in the
            second stage model.
        input_with_ones (bool): Whether to concatenate an extra ones tensor in
            input. Default: True.
        disc_input_with_mask (bool): Whether to add mask as input in
            discriminator. Default: False.
    """

    def __init__(self,
                 *args,
                 stage1_loss_type=('loss_l1_hole', ),
                 stage2_loss_type=('loss_l1_hole', 'loss_gan'),
                 input_with_ones=True,
                 disc_input_with_mask=False,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.stage1_loss_type = stage1_loss_type
        self.stage2_loss_type = stage2_loss_type
        self.input_with_ones = input_with_ones
        self.disc_input_with_mask = disc_input_with_mask
        self.eval_with_metrics = ('metrics' in self.test_cfg) and (
            self.test_cfg['metrics'] is not None)

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
                image. Defaults to False.
            save_path (str, optional): If given a valid str, the results will
                be saved in this path. Defaults to None.
            iteration (int, optional): Iteration number. Defaults to None.

        Returns:
            dict: Contain output results and eval metrics (if have).
        """
        if self.input_with_ones:
            tmp_ones = torch.ones_like(mask)
            input_x = torch.cat([masked_img, tmp_ones, mask], dim=1)
        else:
            input_x = torch.cat([masked_img, mask], dim=1)
        stage1_fake_res, stage2_fake_res = self.generator(input_x)
        fake_img = stage2_fake_res * mask + masked_img * (1. - mask)
        output = dict()
        eval_result = {}
        if self.eval_with_metrics:
            gt_img = kwargs['gt_img']
            data_dict = dict(
                gt_img=gt_img, fake_res=stage2_fake_res, mask=mask)
            for metric_name in self.test_cfg['metrics']:
                if metric_name in ['ssim', 'psnr']:
                    eval_result[metric_name] = self._eval_metrics[metric_name](
                        tensor2img(fake_img, min_max=(-1, 1)),
                        tensor2img(gt_img, min_max=(-1, 1)))
                else:
                    eval_result[metric_name] = self._eval_metrics[metric_name](
                    )(data_dict).item()
            output['eval_result'] = eval_result
        else:
            output['stage1_fake_res'] = stage1_fake_res
            output['stage2_fake_res'] = stage2_fake_res
            output['fake_res'] = stage2_fake_res
            output['fake_img'] = fake_img

        output['meta'] = None if 'meta' not in kwargs else kwargs['meta'][0]

        if save_image:
            assert save_image and save_path is not None, (
                'Save path should be given')
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
            img_list.extend([
                masked_img,
                mask.expand_as(masked_img), stage1_fake_res, stage2_fake_res,
                fake_img
            ])
            img = torch.cat(img_list, dim=3).cpu()
            self.save_visualization(img, osp.join(save_path, filename))
            output['save_img_path'] = osp.abspath(
                osp.join(save_path, filename))

        return output

    def save_visualization(self, img, filename):
        """Save visualization results.

        Args:
            img (torch.Tensor): Tensor with shape of (n, 3, h, w).
            filename (str): Path to save visualization.
        """
        if self.test_cfg.get('img_rerange', True):
            img = (img + 1) / 2
        if self.test_cfg.get('img_bgr2rgb', True):
            img = img[:, [2, 1, 0], ...]
        save_image(img, filename, nrow=1, padding=0)

    def two_stage_loss(self, stage1_data, stage2_data, data_batch):
        """Calculate two-stage loss.

        Args:
            stage1_data (dict): Contain stage1 results.
            stage2_data (dict): Contain stage2 results.
            data_batch (dict): Contain data needed to calculate loss.

        Returns:
            dict: Contain losses with name.
        """
        gt = data_batch['gt_img']
        mask = data_batch['mask']
        masked_img = data_batch['masked_img']

        loss = dict()
        results = dict(
            gt_img=gt.cpu(), mask=mask.cpu(), masked_img=masked_img.cpu())
        # calculate losses for stage1
        if self.stage1_loss_type is not None:
            fake_res = stage1_data['fake_res']
            fake_img = stage1_data['fake_img']
            for type_key in self.stage1_loss_type:
                tmp_loss = self.calculate_loss_with_type(
                    type_key, fake_res, fake_img, gt, mask, prefix='stage1_')
                loss.update(tmp_loss)

        results.update(
            dict(
                stage1_fake_res=stage1_data['fake_res'].cpu(),
                stage1_fake_img=stage1_data['fake_img'].cpu()))

        if self.stage2_loss_type is not None:
            fake_res = stage2_data['fake_res']
            fake_img = stage2_data['fake_img']
            for type_key in self.stage2_loss_type:
                tmp_loss = self.calculate_loss_with_type(
                    type_key, fake_res, fake_img, gt, mask, prefix='stage2_')
                loss.update(tmp_loss)
        results.update(
            dict(
                stage2_fake_res=stage2_data['fake_res'].cpu(),
                stage2_fake_img=stage2_data['fake_img'].cpu()))

        return results, loss

    def calculate_loss_with_type(self,
                                 loss_type,
                                 fake_res,
                                 fake_img,
                                 gt,
                                 mask,
                                 prefix='stage1_'):
        """Calculate multiple types of losses.

        Args:
            loss_type (str): Type of the loss.
            fake_res (torch.Tensor): Direct results from model.
            fake_img (torch.Tensor): Composited results from model.
            gt (torch.Tensor): Ground-truth tensor.
            mask (torch.Tensor): Mask tensor.
            prefix (str, optional): Prefix for loss name.
                Defaults to 'stage1_'.

        Returns:
            dict: Contain loss value with its name.
        """
        loss_dict = dict()
        if loss_type == 'loss_gan':
            if self.disc_input_with_mask:
                disc_input_x = torch.cat([fake_img, mask], dim=1)
            else:
                disc_input_x = fake_img
            g_fake_pred = self.disc(disc_input_x)
            loss_g_fake = self.loss_gan(g_fake_pred, True, is_disc=False)
            loss_dict[prefix + 'loss_g_fake'] = loss_g_fake
        elif 'percep' in loss_type:
            loss_pecep, loss_style = self.loss_percep(fake_img, gt)
            if loss_pecep is not None:
                loss_dict[prefix + loss_type] = loss_pecep
            if loss_style is not None:
                loss_dict[prefix + loss_type[:-6] + 'style'] = loss_style
        elif 'tv' in loss_type:
            loss_tv = self.loss_tv(fake_img, mask=mask)
            loss_dict[prefix + loss_type] = loss_tv
        elif 'l1' in loss_type:
            weight = 1. - mask if 'valid' in loss_type else mask
            loss_l1 = getattr(self, loss_type)(fake_res, gt, weight=weight)
            loss_dict[prefix + loss_type] = loss_l1
        else:
            raise NotImplementedError(
                f'Please check your loss type {loss_type}'
                f' and the config dict in init function. '
                f'We cannot find the related loss function.')

        return loss_dict

    def train_step(self, data_batch, optimizer):
        """Train step function.

        In this function, the inpaintor will finish the train step following
        the pipeline:

            1. get fake res/image
            2. optimize discriminator (if have)
            3. optimize generator

        If `self.train_cfg.disc_step > 1`, the train step will contain multiple
        iterations for optimizing discriminator with different input data and
        only one iteration for optimizing gerator after `disc_step` iterations
        for discriminator.

        Args:
            data_batch (torch.Tensor): Batch of data as input.
            optimizer (dict[torch.optim.Optimizer]): Dict with optimizers for
                generator and discriminator (if have).

        Returns:
            dict: Dict with loss, information for logger, the number of \
                samples and results for visualization.
        """
        log_vars = {}

        gt_img = data_batch['gt_img']
        mask = data_batch['mask']
        masked_img = data_batch['masked_img']

        # get common output from encdec
        if self.input_with_ones:
            tmp_ones = torch.ones_like(mask)
            input_x = torch.cat([masked_img, tmp_ones, mask], dim=1)
        else:
            input_x = torch.cat([masked_img, mask], dim=1)
        stage1_fake_res, stage2_fake_res = self.generator(input_x)
        stage1_fake_img = masked_img * (1. - mask) + stage1_fake_res * mask
        stage2_fake_img = masked_img * (1. - mask) + stage2_fake_res * mask

        # discriminator training step
        # In this version, we only use the results from the second stage to
        # train discriminators, which is a commonly used setting. This can be
        # easily modified to your custom training schedule.
        if self.train_cfg.disc_step > 0:
            set_requires_grad(self.disc, True)
            if self.disc_input_with_mask:
                disc_input_x = torch.cat([stage2_fake_img.detach(), mask],
                                         dim=1)
            else:
                disc_input_x = stage2_fake_img.detach()
            disc_losses = self.forward_train_d(
                disc_input_x, False, is_disc=True)
            loss_disc, log_vars_d = self.parse_losses(disc_losses)
            log_vars.update(log_vars_d)
            optimizer['disc'].zero_grad()
            loss_disc.backward()

            if self.disc_input_with_mask:
                disc_input_x = torch.cat([gt_img, mask], dim=1)
            else:
                disc_input_x = gt_img
            disc_losses = self.forward_train_d(
                disc_input_x, True, is_disc=True)
            loss_disc, log_vars_d = self.parse_losses(disc_losses)
            log_vars.update(log_vars_d)
            loss_disc.backward()

            if self.with_gp_loss:
                # gradient penalty loss should not be used with mask as input
                assert not self.disc_input_with_mask
                loss_d_gp = self.loss_gp(
                    self.disc, gt_img, stage2_fake_img, mask=mask)
                loss_disc, log_vars_d = self.parse_losses(
                    dict(loss_gp=loss_d_gp))
                log_vars.update(log_vars_d)
                loss_disc.backward()

            optimizer['disc'].step()

            self.disc_step_count = (self.disc_step_count +
                                    1) % self.train_cfg.disc_step
            if self.disc_step_count != 0:
                # results contain the data for visualization
                results = dict(
                    gt_img=gt_img.cpu(),
                    masked_img=masked_img.cpu(),
                    fake_res=stage2_fake_res.cpu(),
                    fake_img=stage2_fake_img.cpu())
                outputs = dict(
                    log_vars=log_vars,
                    num_samples=len(data_batch['gt_img'].data),
                    results=results)

                return outputs

        # prepare stage1 results and stage2 results dict for calculating losses
        stage1_results = dict(
            fake_res=stage1_fake_res, fake_img=stage1_fake_img)
        stage2_results = dict(
            fake_res=stage2_fake_res, fake_img=stage2_fake_img)

        # generator (encdec) and refiner training step, results contain the
        # data for visualization
        if self.with_gan:
            set_requires_grad(self.disc, False)
        results, two_stage_losses = self.two_stage_loss(
            stage1_results, stage2_results, data_batch)
        loss_two_stage, log_vars_two_stage = self.parse_losses(
            two_stage_losses)
        log_vars.update(log_vars_two_stage)
        optimizer['generator'].zero_grad()
        loss_two_stage.backward()
        optimizer['generator'].step()

        outputs = dict(
            log_vars=log_vars,
            num_samples=len(data_batch['gt_img'].data),
            results=results)

        return outputs
