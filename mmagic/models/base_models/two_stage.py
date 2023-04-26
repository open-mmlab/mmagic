# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Tuple, Union

import torch
from mmengine.config import Config
from mmengine.optim import OptimWrapperDict

from mmagic.registry import MODELS
from mmagic.utils import SampleList
from ..utils import set_requires_grad
from .one_stage import OneStageInpaintor


@MODELS.register_module()
class TwoStageInpaintor(OneStageInpaintor):
    """Standard two-stage inpaintor with commonly used losses. A two-stage
    inpaintor contains two encoder-decoder style generators to inpaint masked
    regions. Currently, we support these loss types in each of two stage
    inpaintors:

    ['loss_gan', 'loss_l1_hole', 'loss_l1_valid', 'loss_composed_percep',\
     'loss_out_percep', 'loss_tv']
    The `stage1_loss_type` and `stage2_loss_type` should be chosen from these
    loss types.

    Args:
        data_preprocessor (dict): Config of data_preprocessor.
        encdec (dict): Config for encoder-decoder style generator.
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
        stage1_loss_type (tuple[str]): Contains the loss names used in the
            first stage model. Default: ('loss_l1_hole').
        stage2_loss_type (tuple[str]): Contains the loss names used in the
            second stage model. Default: ('loss_l1_hole', 'loss_gan').
        input_with_ones (bool): Whether to concatenate an extra ones tensor in
            input. Default: True.
        disc_input_with_mask (bool): Whether to add mask as input in
            discriminator. Default: False.
    """

    def __init__(
            self,
            data_preprocessor: Union[dict, Config],
            encdec: dict,
            disc: Optional[dict] = None,
            loss_gan: Optional[dict] = None,
            loss_gp: Optional[dict] = None,
            loss_disc_shift: Optional[dict] = None,
            loss_composed_percep: Optional[dict] = None,
            loss_out_percep: bool = False,
            loss_l1_hole: Optional[dict] = None,
            loss_l1_valid: Optional[dict] = None,
            loss_tv: Optional[dict] = None,
            train_cfg: Optional[dict] = None,
            test_cfg: Optional[dict] = None,
            init_cfg: Optional[dict] = None,
            stage1_loss_type: Optional[Sequence[str]] = ('loss_l1_hole', ),
            stage2_loss_type: Optional[Sequence[str]] = ('loss_l1_hole',
                                                         'loss_gan'),
            input_with_ones: bool = True,
            disc_input_with_mask: bool = False):
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

        self.stage1_loss_type = stage1_loss_type
        self.stage2_loss_type = stage2_loss_type
        self.input_with_ones = input_with_ones
        self.disc_input_with_mask = disc_input_with_mask

        if self.train_cfg is not None:
            self.cur_iter = self.train_cfg.start_iter

    def forward_tensor(self, inputs: torch.Tensor, data_samples: SampleList
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward function in tensor mode.

        Args:
            inputs (torch.Tensor): Input tensor.
            data_samples (List[dict]): List of data sample dict.
        Returns:
            dict: Dict contains output results.
        """
        # Pre-process runs in BaseModel.val_step / test_step
        masked_imgs = inputs  # N,3,H,W

        masks = data_samples.mask
        if self.input_with_ones:
            tmp_ones = torch.ones_like(masks)
            input_xs = torch.cat([masked_imgs, tmp_ones, masks], dim=1)
        else:
            input_xs = torch.cat([masked_imgs, masks], dim=1)  # N,4,H,W
        stage1_fake_res, stage2_fake_res = self.generator(input_xs)
        fake_imgs = stage2_fake_res * masks + masked_imgs * (1. - masks)
        return stage2_fake_res, fake_imgs

    def two_stage_loss(self, stage1_data: dict, stage2_data: dict,
                       gt: torch.Tensor, mask: torch.Tensor,
                       masked_img: torch.Tensor) -> Tuple[dict, dict]:
        """Calculate two-stage loss.

        Args:
            stage1_data (dict): Contain stage1 results.
            stage2_data (dict): Contain stage2 results..
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
                                 loss_type: str,
                                 fake_res: torch.Tensor,
                                 fake_img: torch.Tensor,
                                 gt: torch.Tensor,
                                 mask: torch.Tensor,
                                 prefix: Optional[str] = 'stage1_') -> dict:
        """Calculate multiple types of losses.

        Args:
            loss_type (str): Type of the loss.
            fake_res (torch.Tensor): Direct results from model.
            fake_img (torch.Tensor): Composited results from model.
            gt (torch.Tensor): Ground-truth tensor.
            mask (torch.Tensor): Mask tensor.
            prefix (str, optional): Prefix for loss name.
                Defaults to 'stage1\_'. # noqa
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

    def train_step(self, data: List[dict],
                   optim_wrapper: OptimWrapperDict) -> dict:
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
            optim_wrapper['disc'].zero_grad()
            optim_wrapper['disc'].backward(loss_disc)

            if self.disc_input_with_mask:
                disc_input_x = torch.cat([gt_img, mask], dim=1)
            else:
                disc_input_x = gt_img
            disc_losses = self.forward_train_d(
                disc_input_x, True, is_disc=True)
            loss_disc, log_vars_d = self.parse_losses(disc_losses)
            log_vars.update(log_vars_d)
            optim_wrapper['disc'].backward(loss_disc)

            if self.with_gp_loss:
                # gradient penalty loss should not be used with mask as input
                assert not self.disc_input_with_mask
                loss_d_gp = self.loss_gp(
                    self.disc, gt_img, stage2_fake_img, mask=mask)
                loss_disc, log_vars_d = self.parse_losses(
                    dict(loss_gp=loss_d_gp))
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
                    fake_res=stage2_fake_res.cpu(),
                    fake_img=stage2_fake_img.cpu())

                return log_vars

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
            stage1_results, stage2_results, gt_img, mask, masked_img)
        loss_two_stage, log_vars_two_stage = self.parse_losses(
            two_stage_losses)
        log_vars.update(log_vars_two_stage)
        optim_wrapper['generator'].zero_grad()
        optim_wrapper['generator'].backward(loss_two_stage)
        optim_wrapper['generator'].step()

        return log_vars
