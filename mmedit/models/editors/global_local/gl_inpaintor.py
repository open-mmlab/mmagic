# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch

from mmedit.models.base_models import OneStageInpaintor
# from .one_stage import OneStageInpaintor
from mmedit.models.utils import extract_around_bbox, extract_bbox_patch
from mmedit.registry import MODELS
from ...utils import set_requires_grad


@MODELS.register_module()
class GLInpaintor(OneStageInpaintor):
    """Inpaintor for global&local method.

    This inpaintor is implemented according to the paper:
    Globally and Locally Consistent Image Completion

    Importantly, this inpaintor is an example for using custom training
    schedule based on `OneStageInpaintor`.

    The training pipeline of global&local is as following:

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
    :math:`T_D` of theoriginal paper.

    Args:
        generator (dict): Config for encoder-decoder style generator.
        disc (dict): Config for discriminator.
        loss_gan (dict): Config for adversarial loss.
        loss_gp (dict): Config for gradient penalty loss.
        loss_disc_shift (dict): Config for discriminator shift loss.
        loss_composed_percep (dict): Config for perceptural and style loss with
            composed image as input.
        loss_out_percep (dict): Config for perceptural and style loss with
            direct output as input.
        loss_l1_hole (dict): Config for l1 loss in the hole.
        loss_l1_valid (dict): Config for l1 loss in the valid region.
        loss_tv (dict): Config for total variation loss.
        train_cfg (dict): Configs for training scheduler. `disc_step` must be
            contained for indicates the discriminator updating steps in each
            training step.
        test_cfg (dict): Configs for testing scheduler.
        init_cfg (dict, optional): Initialization config dict. Default: None.
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
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)

        if self.train_cfg is not None:
            self.cur_iter = self.train_cfg.start_iter

    def generator_loss(self, fake_res, fake_img, fake_local, gt, mask,
                       masked_img):
        """Forward function in generator training step.

        In this function, we mainly compute the loss items for generator with
        the given (fake_res, fake_img). In general, the `fake_res` is the
        direct output of the generator and the `fake_img` is the composition of
        direct output and ground-truth image.

        Args:
            fake_res (torch.Tensor): Direct output of the generator.
            fake_img (torch.Tensor): Composition of `fake_res` and
                ground-truth image.
            fake_local (torch.Tensor): Local image.
            gt (torch.Tensor): Ground-truth image.
            mask (torch.Tensor): Mask image.
            masked_img (torch.Tensor): Composition of mask image and
                ground-truth image.
        Returns:
            tuple[dict]: A tuple containing two dictionaries. The first one \
                is the result dict, which contains the results computed \
                within this function for visualization. The second one is the \
                loss dict, containing loss items computed in this function.
        """
        loss = dict()

        # if cur_iter <= iter_td, do not calculate adversarial loss
        if self.with_gan and self.cur_iter > self.train_cfg.iter_td:
            g_fake_pred = self.disc((fake_img, fake_local))
            loss_g_fake = self.loss_gan(g_fake_pred, True, False)
            loss['loss_g_fake'] = loss_g_fake

        if self.with_l1_hole_loss:
            loss_l1_hole = self.loss_l1_hole(fake_res, gt, weight=mask)
            loss['loss_l1_hole'] = loss_l1_hole

        if self.with_l1_valid_loss:
            loss_l1_valid = self.loss_l1_valid(fake_res, gt, weight=1. - mask)
            loss['loss_l1_valid'] = loss_l1_valid

        res = dict(
            gt_img=gt.cpu(),
            masked_img=masked_img.cpu(),
            fake_res=fake_res.cpu(),
            fake_img=fake_img.cpu())

        return res, loss

    def train_step(self, data: List[dict], optim_wrapper):
        """Train step function.

        In this function, the inpaintor will finish the train step following
        the pipeline:

        1. get fake res/image
        2. optimize discriminator (if in current schedule)
        3. optimize generator (if in current schedule)

        If ``self.train_cfg.disc_step > 1``, the train step will contain
        multiple iterations for optimizing discriminator with different input
        data and sonly one iteration for optimizing generator after `disc_step`
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
        gt_img = torch.stack([d.gt_img.data
                              for d in data_samples])  # float, [-1,1]
        mask = torch.stack([d.mask.data for d in data_samples])  # uint8, {0,1}
        mask = mask.float()

        bbox_tensor = torch.tensor([d.mask_bbox for d in data_samples])  # int

        input_x = torch.cat([masked_img, mask], dim=1)
        fake_res = self.generator(input_x)
        fake_img = gt_img * (1. - mask) + fake_res * mask

        fake_local, bbox_new = extract_around_bbox(fake_img, bbox_tensor,
                                                   self.train_cfg.local_size)
        gt_local = extract_bbox_patch(bbox_new, gt_img)
        fake_gt_local = torch.cat([fake_local, gt_local], dim=2)

        # if cur_iter > iter_tc, update discriminator
        if (self.train_cfg.disc_step > 0
                and self.cur_iter > self.train_cfg.iter_tc):
            # set discriminator requires_grad as True
            set_requires_grad(self.disc, True)

            fake_data = (fake_img.detach(), fake_local.detach())
            real_data = (gt_img, gt_local)
            disc_losses = self.forward_train_d(fake_data, False, True)
            loss_disc, log_vars_d = self.parse_losses(disc_losses)
            log_vars.update(log_vars_d)
            optim_wrapper['disc'].zero_grad()
            optim_wrapper['disc'].backward(loss_disc)

            disc_losses = self.forward_train_d(real_data, True, True)
            loss_disc, log_vars_d = self.parse_losses(disc_losses)
            log_vars.update(log_vars_d)
            optim_wrapper['disc'].backward(loss_disc)
            optim_wrapper['disc'].step()
            self.disc_step_count = (self.disc_step_count +
                                    1) % self.train_cfg.disc_step

            # if cur_iter <= iter_td, do not update generator
            if (self.disc_step_count != 0
                    or self.cur_iter <= self.train_cfg.iter_td):
                results = dict(
                    gt_img=gt_img.cpu(),
                    masked_img=masked_img.cpu(),
                    fake_res=fake_res.cpu(),
                    fake_img=fake_img.cpu(),
                    fake_gt_local=fake_gt_local.cpu())
                # outputs = dict(**log_vars,**results)

                self.cur_iter += 1
                return log_vars

        # set discriminators requires_grad as False to avoid extra computation.
        set_requires_grad(self.disc, False)
        # update generator
        if (self.cur_iter <= self.train_cfg.iter_tc
                or self.cur_iter > self.train_cfg.iter_td):
            results, g_losses = self.generator_loss(fake_res, fake_img,
                                                    fake_local, gt_img, mask,
                                                    masked_img)
            loss_g, log_vars_g = self.parse_losses(g_losses)
            log_vars.update(log_vars_g)
            optim_wrapper['generator'].zero_grad()
            optim_wrapper['generator'].backward(loss_g)
            optim_wrapper['generator'].step()

            results.update(fake_gt_local=fake_gt_local.cpu())
            # outputs = dict(**log_vars,**results)

        self.cur_iter += 1

        return log_vars
