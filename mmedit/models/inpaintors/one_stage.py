import torch

from ..base import BaseModel
from ..builder import build_backbone, build_component, build_loss
from ..registry import MODELS


@MODELS.register_module
class OneStageInpaintor(BaseModel):
    """Standard one-stage inpaintor with commonly used losses.

    An inpaintor must contain an encoder-decoder style generator to
    inpaint masked regions. A discriminator will be adopted when
    adversarial training is needed.

    In this class, we provide a common interface for inpaintors.
    For other inpaintors, only some funcs may be modified to fit the
    input style or training schedule.

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
        super(OneStageInpaintor, self).__init__()
        self.with_l1_hole_loss = loss_l1_hole is not None
        self.with_l1_valid_loss = loss_l1_valid is not None
        self.with_tv_loss = loss_tv is not None
        self.with_composed_percep_loss = loss_composed_percep is not None
        self.with_out_percep_loss = loss_out_percep
        self.with_gan = disc is not None and loss_gan is not None
        self.with_gp_loss = loss_gp is not None
        self.with_disc_shift_loss = loss_disc_shift is not None
        self.is_train = train_cfg is not None
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.generator = build_backbone(encdec)

        # build loss modules
        if self.with_gan:
            self.disc = build_component(disc)
            self.loss_gan = build_loss(loss_gan)

        if self.with_l1_hole_loss:
            self.loss_l1_hole = build_loss(loss_l1_hole)

        if self.with_l1_valid_loss:
            self.loss_l1_valid = build_loss(loss_l1_valid)

        if self.with_composed_percep_loss:
            self.loss_percep = build_loss(loss_composed_percep)

        if self.with_gp_loss:
            self.loss_gp = build_loss(loss_gp)

        if self.with_disc_shift_loss:
            self.loss_disc_shift = build_loss(loss_disc_shift)

        if self.with_tv_loss:
            self.loss_tv = build_loss(loss_tv)

        self.disc_step_count = 0
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        self.generator.init_weights(pretrained=pretrained)
        if self.with_gan:
            self.disc.init_weights(pretrained=pretrained)

    def forward_train(self, x):
        raise NotImplementedError('This interface should not be used in '
                                  'current training schedule. Please use '
                                  '`train_step` for training.')

    def forward_train_d(self, data_batch, is_real, is_disc):
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
        loss_ = self.loss_gan(pred, is_real, is_disc)

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
            dict: Contains the results computed within this function for
                visualization.
            dict: Contains the loss items computed in this function.
        """
        gt = data_batch['gt_img']
        mask = data_batch['mask']
        masked_img = data_batch['masked_img']

        loss = dict()

        if self.with_gan:
            with torch.no_grad():
                g_fake_pred = self.disc(fake_img)
            loss_g_fake = self.loss_gan(g_fake_pred, True, is_disc=False)
            loss['loss_g_fake'] = loss_g_fake

        if self.with_l1_hole_loss:
            loss_l1_hole = self.loss_l1_hole(fake_res, gt, weight=mask)
            loss['loss_l1_hole'] = loss_l1_hole

        if self.with_l1_valid_loss:
            loss_loss_l1_valid = self.loss_l1_valid(
                fake_res, gt, weight=1. - mask)
            loss['loss_l1_valid'] = loss_loss_l1_valid

        if self.with_composed_percep_loss:
            loss_pecep, loss_style = self.loss_percep(fake_img, gt)
            if loss_pecep is not None:
                loss['loss_composed_percep'] = loss_pecep
            if loss_style is not None:
                loss['loss_composed_style'] = loss_style

        if self.with_out_percep_loss:
            loss_out_percep, loss_out_style = self.loss_percep(fake_res, gt)
            if loss_out_percep is not None:
                loss['loss_out_percep'] = loss_out_percep
            if loss_out_style is not None:
                loss['loss_out_style'] = loss_out_style

        if self.with_tv_loss:
            loss_tv = self.loss_tv(fake_img, mask=mask)
            loss['loss_tv'] = loss_tv

        res = dict(
            gt_img=gt.cpu(),
            masked_img=masked_img.cpu(),
            fake_res=fake_res.cpu(),
            fake_img=fake_img.cpu())

        return res, loss

    def forward_test(self, data_batch):
        gt_img = data_batch['gt_img']
        mask = data_batch['mask']
        masked_img = data_batch['masked_img']

        input_x = torch.cat([masked_img, mask], dim=1)

        fake_res = self.generator(input_x)
        fake_img = fake_res * mask + gt_img * (1. - mask)
        output = dict(fake_res=fake_res, fake_img=fake_img)
        return output

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
            dict: Dict with loss, information for logger, the number of samples
                and results for visualization.
        """
        log_vars = {}

        gt_img = data_batch['gt_img']
        mask = data_batch['mask']
        masked_img = data_batch['masked_img']

        # get common output from encdec
        input_x = torch.cat([masked_img, mask], dim=1)
        fake_res = self.generator(input_x)
        fake_img = gt_img * (1. - mask) + fake_res * mask

        # discriminator training step
        if self.train_cfg.disc_step > 0:

            disc_losses = self.forward_train_d(
                fake_img.detach(), False, is_disc=True)
            loss_disc, log_vars_d = self.parse_losses(disc_losses)
            log_vars.update(log_vars_d)
            optimizer['disc'].zero_grad()
            loss_disc.backward()

            disc_losses = self.forward_train_d(gt_img, True, is_disc=True)
            loss_disc, log_vars_d = self.parse_losses(disc_losses)
            log_vars.update(log_vars_d)
            loss_disc.backward()

            if self.with_gp_loss:
                loss_d_gp = self.loss_gp(self.disc, gt_img, fake_img)
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

    def val_step(self, data_batch, **kwargs):
        output = self.forward_test(data_batch, **kwargs)

        return output
