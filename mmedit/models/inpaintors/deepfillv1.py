import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel

from ..common import extract_around_bbox, extract_bbox_patch, set_requires_grad
from ..registry import MODELS
from .two_stage import TwoStageInpaintor


@MODELS.register_module()
class DeepFillv1Inpaintor(TwoStageInpaintor):

    def get_module(self, model, module_name):
        """Get an inner module from model.

        Since we will wrapper DDP for some model, we have to judge whether the
        module can be indexed directly.

        Args:
            model (nn.Module): This model may wrapped with DDP or not.
            module_name (str): The name of specific module.

        Return:
            nn.Module: Returned sub module.
        """
        if isinstance(model, (DataParallel, DistributedDataParallel)):
            return getattr(model.module, module_name)
        else:
            return getattr(model, module_name)

    def forward_train_d(self, data_batch, is_real, is_disc):
        """Forward function in discriminator training step.

        In this function, we modify the default implementation with only one
        discriminator. In DeepFillv1 model, they use two separated
        discriminators for global and local consistency.

        Args:
            data (torch.Tensor): Batch of real data or fake data.
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
                Defaults to 'stage1_'.
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
        bbox_tensor = data_batch['mask_bbox']

        # get common output from encdec
        if self.input_with_ones:
            tmp_ones = torch.ones_like(mask)
            input_x = torch.cat([masked_img, tmp_ones, mask], dim=1)
        else:
            input_x = torch.cat([masked_img, mask], dim=1)
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
            optimizer['disc'].zero_grad()
            loss_disc.backward()

            disc_losses = self.forward_train_d(real_data, True, is_disc=True)
            loss_disc, log_vars_d = self.parse_losses(disc_losses)
            log_vars.update(log_vars_d)
            loss_disc.backward()

            if self.with_gp_loss:
                loss_gp_global = self.loss_gp(
                    self.get_module(self.disc, 'global_disc'),
                    gt_img,
                    stage2_fake_img,
                    mask=mask)
                loss_gp_local = self.loss_gp(
                    self.get_module(self.disc, 'local_disc'), gt_local,
                    stage2_fake_local)
                loss_disc, log_vars_d = self.parse_losses(
                    dict(
                        loss_gp_global=loss_gp_global,
                        loss_gp_local=loss_gp_local))
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
                    stage1_fake_res=stage1_fake_res.cpu(),
                    stage1_fake_img=stage1_fake_img.cpu(),
                    stage2_fake_res=stage2_fake_res.cpu(),
                    stage2_fake_img=stage2_fake_img.cpu(),
                    fake_gt_local=fake_gt_local.cpu(),
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
            fake_res=stage2_fake_res,
            fake_img=stage2_fake_img,
            fake_local=stage2_fake_local)

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

        results['fake_gt_local'] = fake_gt_local.cpu()
        outputs = dict(
            log_vars=log_vars,
            num_samples=len(data_batch['gt_img'].data),
            results=results)

        return outputs
