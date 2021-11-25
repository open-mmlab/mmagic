import torch

from ..common import extract_around_bbox, extract_bbox_patch, set_requires_grad
from ..registry import MODELS
from .one_stage import OneStageInpaintor


@MODELS.register_module()
class AOTInpaintor(OneStageInpaintor):
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
        pretrained (str): Path for pretrained model. Default None.
    """

    def __init__(self,
                 encdec,
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
                 pretrained=None):
        super().__init__(
            encdec,
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
            pretrained=pretrained)

        if self.train_cfg is not None:
            self.cur_iter = self.train_cfg.start_iter

    def forward_train_d(self, data_batch, is_real, is_disc, **kwargs):
        """Forward function in discriminator training step.

        In this function, we compute the prediction for each data batch (real
        or fake). Meanwhile, the standard gan loss will be computed with
        several proposed losses fro stable training.

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
        
        pred = self.disc(data_batch)
        loss_ = self.loss_gan(pred, is_real, is_disc, **kwargs)
        

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
            tuple(dict): Dict contains the results computed within this \
                function for visualization and dict contains the loss items \
                computed in this function.
        """
        gt = data_batch['gt_img']
        mask = data_batch['mask']
        masked_img = data_batch['masked_img']

        loss = dict()
        
        if self.with_gan:
            pred = self.disc(fake_res)
            loss_g_fake = self.loss_gan(pred, True, False, mask=mask)
            loss['loss_g_fake'] = loss_g_fake
        
        if self.with_l1_valid_loss:
            loss_loss_l1_valid = self.loss_l1_valid(
                fake_res, gt, weight=1. - mask)
            loss['loss_l1_valid'] = loss_loss_l1_valid
        
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

    def train_step(self, data_batch, optimizer):
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
        masked_img = masked_img.float() + mask

        
        '''
        import cv2
        img_vis = gt_img[0].cpu().detach().numpy().transpose(1, 2, 0)
        cv2.imshow('vis0.png', img_vis)
        cv2.waitKey(0)
        '''

        input_x = torch.cat([masked_img, mask], dim=1)
        fake_res = self.generator(input_x)
        fake_img = gt_img * (1. - mask) + fake_res * mask
        '''
        import cv2
        img_vis = fake_res[0].cpu().detach().numpy().transpose(1, 2, 0)
        cv2.imshow('vis0.png', img_vis)
        cv2.waitKey(0)
        '''
        # reconstruction losses
        results, g_losses = self.generator_loss(fake_res, fake_img, data_batch)
        loss_g, log_vars_g = self.parse_losses(g_losses)
        log_vars.update(log_vars_g)

        # adversarial loss
        set_requires_grad(self.disc, True)
        fake_data = fake_img.detach()
        real_data = gt_img
        disc_losses_real = self.forward_train_d(
            real_data, True, True, mask=mask)
        disc_losses_fake = self.forward_train_d(
            fake_data, False, True, mask=mask)
        disc_losses_ = disc_losses_real['real_loss'] + disc_losses_fake['fake_loss']
        disc_losses = dict(disc_losses=disc_losses_)
        loss_disc, log_vars_d = self.parse_losses(disc_losses)
        # log_vars.update(log_vars_d)

        # backprop
        optimizer['generator'].zero_grad()
        optimizer['disc'].zero_grad()
        loss_g.backward()
        loss_disc.backward()
        optimizer['generator'].step()
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

        outputs = dict(
            log_vars=log_vars,
            num_samples=len(data_batch['gt_img'].data),
            results=results)

        return outputs

        '''
        

         # discriminator training step
        if self.train_cfg.disc_step > 0:
            set_requires_grad(self.disc, True)
            
            fake_data = fake_img.detach()
            real_data = gt_img
            disc_losses_real = self.forward_train_d(
                real_data, True, True, mask=mask)
            disc_losses_fake = self.forward_train_d(
                fake_data, False, True, mask=mask)
            disc_losses_ = disc_losses_real['real_loss'] + disc_losses_fake['fake_loss']
            disc_losses = dict(disc_losses=disc_losses_)
            
            loss_disc, log_vars_d = self.parse_losses(disc_losses)
            log_vars.update(log_vars_d)
            optimizer['disc'].zero_grad()
            loss_disc.backward()

            if self.with_gp_loss:
                loss_d_gp = self.loss_gp(
                    self.disc, gt_img, fake_img, mask=mask)
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
                    fake_res=fake_res.cpu(),
                    fake_img=fake_img.cpu())
                outputs = dict(
                    log_vars=log_vars,
                    num_samples=len(data_batch['gt_img'].data),
                    results=results)

                return outputs

        # generator (encdec) training step, results contain the data
        # for visualization
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

'''