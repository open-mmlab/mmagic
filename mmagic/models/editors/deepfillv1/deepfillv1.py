# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch

from mmagic.models.base_models import TwoStageInpaintor
from mmagic.models.utils import extract_around_bbox, extract_bbox_patch
from mmagic.registry import MODELS
from ...utils import set_requires_grad


@MODELS.register_module()
class DeepFillv1Inpaintor(TwoStageInpaintor):
    """Inpaintor for deepfillv1 method.

    This inpaintor is implemented according to the paper:
    Generative image inpainting with contextual attention

    Importantly, this inpaintor is an example for using custom training
    schedule based on `TwoStageInpaintor`.

    The training pipeline of deepfillv1 is as following:

    .. code-block:: python

        if cur_iter < iter_tc:
            update generator with only l1 loss
        else:
            update discriminator
            if cur_iter > iter_td:
                update generator with l1 loss and adversarial loss

    The new attribute `cur_iter` is added for recording current number of
    iteration. The `train_cfg` contains the setting of the training schedule:

    .. code-block:: python

        train_cfg = dict(
            start_iter=0,
            disc_step=1,
            iter_tc=90000,
            iter_td=100000
        )

    `iter_tc` and `iter_td` correspond to the notation :math:`T_C` and
    :math:`T_D` of the original paper.

    Args:
        generator (dict): Config for encoder-decoder style generator.
        disc (dict): Config for discriminator.
        loss_gan (dict): Config for adversarial loss.
        loss_gp (dict): Config for gradient penalty loss.
        loss_disc_shift (dict): Config for discriminator shift loss.
        loss_composed_percep (dict): Config for perceptual and style loss with
            composed image as input.
        loss_out_percep (dict): Config for perceptual and style loss with
            direct output as input.
        loss_l1_hole (dict): Config for l1 loss in the hole.
        loss_l1_valid (dict): Config for l1 loss in the valid region.
        loss_tv (dict): Config for total variation loss.
        train_cfg (dict): Configs for training scheduler. `disc_step` must be
            contained for indicates the discriminator updating steps in each
            training step.
        test_cfg (dict): Configs for testing scheduler.
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self,
                 data_preprocessor: dict,
                 encdec: dict,
                 disc=None,
                 loss_gan=None,
                 loss_gp=None,
                 loss_disc_shift=None,
                 loss_composed_percep=None,
                 loss_out_percep=False,
                 loss_l1_hole=None,
                 loss_l1_valid=None,
                 loss_tv=None,
                 stage1_loss_type=None,
                 stage2_loss_type=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg: Optional[dict] = None):
        super().__init__(
            data_preprocessor=data_preprocessor,
            encdec=encdec,
            disc=disc,
            loss_gan=loss_gan,
            loss_gp=loss_gp,
            loss_disc_shift=loss_disc_shift,
            loss_composed_percep=loss_composed_percep,
            loss_out_percep=loss_out_percep,
            loss_l1_hole=loss_l1_hole,
            loss_l1_valid=loss_l1_valid,
            loss_tv=loss_tv,
            stage1_loss_type=stage1_loss_type,
            stage2_loss_type=stage2_loss_type,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)

        if self.train_cfg is not None:
            self.cur_iter = self.train_cfg.start_iter

    def forward_train_d(self, data_batch, is_real, is_disc):
        """Forward function in discriminator training step.

        In this function, we modify the default implementation with only one
        discriminator. In DeepFillv1 model, they use two separated
        discriminators for global and local consistency.

        Args:
            data_batch (torch.Tensor): Batch of real data or fake data.
            is_real (bool): If True, the gan loss will regard this batch as
                real data. Otherwise, the gan loss will regard this batch as
                fake data.
            is_disc (bool): If True, this function is called in discriminator
                training step. Otherwise, this function is called in generator
                training step. This will help us to compute different types of
                adversarial loss, like LSGAN.

        Returns:
            dict: Contains the loss items computed in this function.
        """
        global_pred, local_pred = self.disc(data_batch)
        loss_global = self.loss_gan(global_pred, is_real, is_disc)
        loss_local = self.loss_gan(local_pred, is_real, is_disc)

        if is_real:
            loss = dict(
                real_loss_global=loss_global, real_loss_local=loss_local)
        else:
            loss = dict(
                fake_loss_global=loss_global, fake_loss_local=loss_local)

        if self.with_disc_shift_loss:
            loss_d_shift_global = self.loss_disc_shift(loss_global)
            loss_d_shift_local = self.loss_disc_shift(loss_local)
            # 0.5 for average the fake and real data
            loss.update(loss_disc_shift_global=loss_d_shift_global * 0.5)
            loss.update(loss_disc_shift_local=loss_d_shift_local * 0.5)

        return loss

    def two_stage_loss(self, stage1_data, stage2_data, gt, mask, masked_img):
        """Calculate two-stage loss.

        Args:
            stage1_data (dict): Contain stage1 results.
            stage2_data (dict): Contain stage2 results.
            gt (torch.Tensor): Ground-truth image.
            mask (torch.Tensor): Mask image.
            masked_img (torch.Tensor): Composition of mask image and
                ground-truth image.
        Returns:
            tuple(dict): Dict contains the results computed within this \
                function for visualization and dict contains the loss items \
                computed in this function.
        """
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
            fake_local = stage2_data['fake_local']
            for type_key in self.stage2_loss_type:
                tmp_loss = self.calculate_loss_with_type(
                    type_key,
                    fake_res,
                    fake_img,
                    gt,
                    mask,
                    prefix='stage2_',
                    fake_local=fake_local)
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
                                 prefix='stage1_',
                                 fake_local=None):
        """Calculate multiple types of losses.

        Args:
            loss_type (str): Type of the loss.
            fake_res (torch.Tensor): Direct results from model.
            fake_img (torch.Tensor): Composited results from model.
            gt (torch.Tensor): Ground-truth tensor.
            mask (torch.Tensor): Mask tensor.
            prefix (str, optional): Prefix for loss name.
                Defaults to 'stage1\_'. # noqa
            fake_local (torch.Tensor, optional): Local results from model.
                Defaults to None.

        Returns:
            dict: Contain loss value with its name.
        """
        loss_dict = dict()
        if loss_type == 'loss_gan':
            g_fake_global_pred, g_fake_local_pred = self.disc(
                (fake_img, fake_local))
            loss_g_fake_global = self.loss_gan(
                g_fake_global_pred, True, is_disc=False)
            loss_g_fake_local = self.loss_gan(
                g_fake_local_pred, True, is_disc=False)
            loss_dict[prefix +
                      'loss_g_fake'] = loss_g_fake_global + loss_g_fake_local
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
                ' and the config dict in init function. '
                'We cannot find the related loss function.')

        return loss_dict

    def train_step(self, data: List[dict], optim_wrapper):
        """Train step function.

        In this function, the inpaintor will finish the train step following
        the pipeline:

            1. get fake res/image
            2. optimize discriminator (if have)
            3. optimize generator

        If `self.train_cfg.disc_step > 1`, the train step will contain multiple
        iterations for optimizing discriminator with different input data and
        only one iteration for optimizing generator after `disc_step`
        iterations for discriminator.

        Args:
            data (List[dict]): Batch of data as input.
            optim_wrapper (dict[torch.optim.Optimizer]): Dict with optimizers
                for generator and discriminator (if have).

        Returns:
            dict: Dict with loss, information for logger, the number of \
                samples and results for visualization.
        """
        data = self.data_preprocessor(data, True)
        batch_inputs, data_samples = data['inputs'], data['data_samples']
        log_vars = {}

        masked_img = batch_inputs  # float
        gt_img = data_samples.gt_img
        mask = data_samples.mask
        mask = mask.float()

        # PyTorch 2.0 could not compile 'data_samples.mask_bbox'
        # bbox_tensor = torch.LongTensor(data_samples.mask_bbox)
        bbox_tensor = torch.LongTensor(data_samples.metainfo['mask_bbox'])

        # get common output from encdec
        # input with ones
        tmp_ones = torch.ones_like(mask)
        input_x = torch.cat([masked_img, tmp_ones, mask], dim=1)
        stage1_fake_res, stage2_fake_res = self.generator(input_x)
        stage1_fake_img = masked_img * (1. - mask) + stage1_fake_res * mask
        stage2_fake_img = masked_img * (1. - mask) + stage2_fake_res * mask

        stage2_fake_local, bbox_new = extract_around_bbox(
            stage2_fake_img, bbox_tensor, self.train_cfg.local_size)
        gt_local = extract_bbox_patch(bbox_new, gt_img)
        fake_gt_local = torch.cat([stage2_fake_local, gt_local], dim=2)
        # discriminator training step
        # In this version, we only use the results from the second stage to
        # train discriminators, which is a commonly used setting. This can be
        # easily modified to your custom training schedule.
        if self.train_cfg.disc_step > 0 and self.with_gan:
            set_requires_grad(self.disc, True)
            fake_data = (stage2_fake_img.detach(), stage2_fake_local.detach())
            real_data = (gt_img, gt_local)

            disc_losses = self.forward_train_d(fake_data, False, is_disc=True)
            loss_disc, log_vars_d = self.parse_losses(disc_losses)
            log_vars.update(log_vars_d)
            optim_wrapper['disc'].zero_grad()
            optim_wrapper['disc'].backward(loss_disc)

            disc_losses = self.forward_train_d(real_data, True, is_disc=True)
            loss_disc, log_vars_d = self.parse_losses(disc_losses)
            log_vars.update(log_vars_d)
            optim_wrapper['disc'].backward(loss_disc)

            if self.with_gp_loss:
                if hasattr(self.disc, 'module'):
                    global_disc = self.disc.module.global_disc
                    local_disc = self.disc.module.local_disc
                else:
                    global_disc = self.disc.global_disc
                    local_disc = self.disc.local_disc

                loss_gp_global = self.loss_gp(
                    global_disc, gt_img, stage2_fake_img, mask=mask)
                loss_gp_local = self.loss_gp(local_disc, gt_local,
                                             stage2_fake_local)
                loss_disc, log_vars_d = self.parse_losses(
                    dict(
                        loss_gp_global=loss_gp_global,
                        loss_gp_local=loss_gp_local))
                log_vars.update(log_vars_d)
                optim_wrapper['disc'].backward(loss_disc)

            optim_wrapper['disc'].step()

            self.disc_step_count = (self.disc_step_count +
                                    1) % self.train_cfg.disc_step
            if self.disc_step_count != 0:
                # results contain the data for visualization
                results = dict(
                    gt_img=gt_img.cpu(),
                    masked_img=masked_img.cpu(),
                    stage1_fake_res=stage1_fake_res.cpu(),
                    stage1_fake_img=stage1_fake_img.cpu(),
                    stage2_fake_res=stage2_fake_res.cpu(),
                    stage2_fake_img=stage2_fake_img.cpu(),
                    fake_gt_local=fake_gt_local.cpu(),
                    fake_res=stage2_fake_res.cpu(),
                    fake_img=stage2_fake_img.cpu())

                return log_vars

        # prepare stage1 results and stage2 results dict for calculating losses
        stage1_results = dict(
            fake_res=stage1_fake_res, fake_img=stage1_fake_img)
        stage2_results = dict(
            fake_res=stage2_fake_res,
            fake_img=stage2_fake_img,
            fake_local=stage2_fake_local)

        # generator (encdec) and refiner training step, results contain the
        # data for visualization
        if self.with_gan:
            set_requires_grad(self.disc, False)
        results, two_stage_losses = self.two_stage_loss(
            stage1_results, stage2_results, gt_img, mask, masked_img)
        loss_two_stage, log_vars_two_stage = self.parse_losses(
            two_stage_losses)
        log_vars.update(log_vars_two_stage)
        optim_wrapper['generator'].zero_grad()
        optim_wrapper['generator'].backward(loss_two_stage)
        optim_wrapper['generator'].step()

        results['fake_gt_local'] = fake_gt_local.cpu()

        return log_vars
