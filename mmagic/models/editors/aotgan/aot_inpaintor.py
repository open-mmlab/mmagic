# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch

from mmagic.models.base_models import OneStageInpaintor
from mmagic.registry import MODELS
from ...utils import set_requires_grad


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
            data_batch (torch.Tensor): Batch of real data or fake data.
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

    def generator_loss(self, fake_res, fake_img, gt, mask, masked_img):
        """Forward function in generator training step.

        In this function, we mainly compute the loss items for generator with
        the given (fake_res, fake_img). In general, the `fake_res` is the
        direct output of the generator and the `fake_img` is the composition of
        direct output and ground-truth image.

        Args:
            fake_res (torch.Tensor): Direct output of the generator.
            fake_img (torch.Tensor): Composition of `fake_res` and
                ground-truth image.
            gt (torch.Tensor): Ground-truth image.
            mask (torch.Tensor): Mask image.
            masked_img (torch.Tensor): Composition of mask image and
                ground-truth image.

        Returns:
            tuple(dict): Dict contains the results computed within this
                function for visualization and dict contains the loss items
                computed in this function.
        """
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

    def forward_tensor(self, inputs, data_samples):
        """Forward function in tensor mode.

        Args:
            inputs (torch.Tensor): Input tensor.
            data_samples (List[dict]): List of data sample dict.

        Returns:
            tuple: Direct output of the generator and composition of `fake_res`
                and ground-truth image.
        """
        # Pre-process runs in BaseModel.val_step / test_step
        masks = data_samples.mask

        masked_imgs = inputs  # N,3,H,W
        masked_imgs = masked_imgs.float() + masks

        input_xs = torch.cat([masked_imgs, masks], dim=1)  # N,4,H,W
        fake_reses = self.generator(input_xs)
        fake_imgs = fake_reses * masks + masked_imgs * (1. - masks)
        return fake_reses, fake_imgs

    def train_step(self, data: List[dict], optim_wrapper):
        """Train step function.

        In this function, the inpaintor will finish the train step following
        the pipeline:
        1. get fake res/image
        2. compute reconstruction losses for generator
        3. compute adversarial loss for discriminator
        4. optimize generator
        5. optimize discriminator

        Args:
            data (List[dict]): Batch of data as input.
            optim_wrapper (dict[torch.optim.Optimizer]): Dict with optimizers
                for generator and discriminator (if have).

        Returns:
            dict: Dict with loss, information for logger, the number of
                samples and results for visualization.
        """
        data = self.data_preprocessor(data, True)
        batch_inputs, data_samples = data['inputs'], data['data_samples']
        log_vars = {}

        # prepare data for training
        gt_img = data_samples.gt_img
        mask = data_samples.mask
        mask = mask.float()

        masked_img = batch_inputs
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
            optim_wrapper['disc'].backward(loss_disc)
            optim_wrapper['disc'].step()
            optim_wrapper['disc'].zero_grad()
            log_vars.update(log_vars_d)

            self.disc_step_count = (self.disc_step_count +
                                    1) % self.train_cfg.disc_step
            if self.disc_step_count != 0:
                return log_vars

        # generator (encdec) training step, results contain the data
        # for visualization
        if self.with_gan:
            set_requires_grad(self.disc, False)
        results, g_losses = self.generator_loss(fake_res, fake_img, gt_img,
                                                mask, masked_img)
        loss_g, log_vars_g = self.parse_losses(g_losses)
        log_vars.update(log_vars_g)

        optim_wrapper['generator'].backward(loss_g)
        optim_wrapper['generator'].step()
        optim_wrapper['generator'].zero_grad()

        return log_vars
