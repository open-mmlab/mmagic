import torch

from ..common import extract_around_bbox, extract_bbox_patch, set_requires_grad
from ..registry import MODELS
from .one_stage import OneStageInpaintor


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

    `iter_tc` and `iter_td` correspond to the noation :math:`T_C` and
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
        super(GLInpaintor, self).__init__(
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

    def generator_loss(self, fake_res, fake_img, fake_local, data_batch):
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
            tuple[dict]: A tuple containing two dictionaries. The first one \
                is the result dict, which contains the results computed \
                within this function for visualization. The second one is the \
                loss dict, containing loss items computed in this function.
        """
        gt = data_batch['gt_img']
        mask = data_batch['mask']
        masked_img = data_batch['masked_img']

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

    def train_step(self, data_batch, optimizer):
        """Train step function.

        In this function, the inpaintor will finish the train step following
        the pipeline:

        1. get fake res/image
        2. optimize discriminator (if in current schedule)
        3. optimzie generator (if in current schedule)

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
        bbox_tensor = data_batch['mask_bbox']

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
            optimizer['disc'].zero_grad()
            loss_disc.backward()

            disc_losses = self.forward_train_d(real_data, True, True)
            loss_disc, log_vars_d = self.parse_losses(disc_losses)
            log_vars.update(log_vars_d)
            loss_disc.backward()
            optimizer['disc'].step()
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
                outputs = dict(
                    log_vars=log_vars,
                    num_samples=len(data_batch['gt_img'].data),
                    results=results)

                self.cur_iter += 1
                return outputs

        # set discriminators requires_grad as False to avoid extra computation.
        set_requires_grad(self.disc, False)
        # update generator
        if (self.cur_iter <= self.train_cfg.iter_tc
                or self.cur_iter > self.train_cfg.iter_td):
            results, g_losses = self.generator_loss(fake_res, fake_img,
                                                    fake_local, data_batch)
            loss_g, log_vars_g = self.parse_losses(g_losses)
            log_vars.update(log_vars_g)
            optimizer['generator'].zero_grad()
            loss_g.backward()
            optimizer['generator'].step()

            results.update(fake_gt_local=fake_gt_local.cpu())
            outputs = dict(
                log_vars=log_vars,
                num_samples=len(data_batch['gt_img'].data),
                results=results)

        self.cur_iter += 1

        return outputs
