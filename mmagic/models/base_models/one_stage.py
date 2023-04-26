# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
from mmengine.config import Config
from mmengine.model import BaseModel
from mmengine.optim import OptimWrapperDict

from mmagic.registry import MODELS
from mmagic.structures import DataSample
from mmagic.utils import SampleList
from ..utils import set_requires_grad

FORWARD_RETURN_TYPE = Union[dict, torch.Tensor,
                            Tuple[torch.Tensor, torch.Tensor], SampleList]


@MODELS.register_module()
class OneStageInpaintor(BaseModel):
    """Standard one-stage inpaintor with commonly used losses.

    An inpaintor must contain an encoder-decoder style generator to
    inpaint masked regions. A discriminator will be adopted when
    adversarial training is needed.

    In this class, we provide a common interface for inpaintors.
    For other inpaintors, only some funcs may be modified to fit the
    input style or training schedule.

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
    """

    def __init__(self,
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
                 init_cfg: Optional[dict] = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
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

        self.generator = MODELS.build(encdec)

        # build loss modules
        if self.with_gan:
            self.disc = MODELS.build(disc)
            self.loss_gan = MODELS.build(loss_gan)

        if self.with_l1_hole_loss:
            self.loss_l1_hole = MODELS.build(loss_l1_hole)

        if self.with_l1_valid_loss:
            self.loss_l1_valid = MODELS.build(loss_l1_valid)

        if self.with_composed_percep_loss:
            self.loss_percep = MODELS.build(loss_composed_percep)

        if self.with_gp_loss:
            self.loss_gp = MODELS.build(loss_gp)

        if self.with_disc_shift_loss:
            self.loss_disc_shift = MODELS.build(loss_disc_shift)

        if self.with_tv_loss:
            self.loss_tv = MODELS.build(loss_tv)

        self.disc_step_count = 0

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[SampleList],
                mode: str = 'tensor') -> FORWARD_RETURN_TYPE:
        """Forward function.

        Args:
            inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (List[BaseDataElement], optional):
                data samples collated by :attr:`data_preprocessor`.
            mode (str): mode should be one of ``loss``, ``predict`` and
                ``tensor``. Default: 'tensor'.

                - ``loss``: Called by ``train_step`` and return loss ``dict``
                  used for logging
                - ``predict``: Called by ``val_step`` and ``test_step``
                  and return list of ``BaseDataElement`` results used for
                  computing metric.
                - ``tensor``: Called by custom use to get ``Tensor`` type
                  results.

        Returns:
            ForwardResults:

                - If ``mode == loss``, return a ``dict`` of loss tensor used
                  for backward and logging.
                - If ``mode == predict``, return a ``list`` of
                  :obj:`BaseDataElement` for computing metric
                  and getting inference result.
                - If ``mode == tensor``, return a tensor or ``tuple`` of tensor
                  or ``dict`` or tensor for custom use.
        """
        if mode == 'tensor':
            raw = self.forward_tensor(inputs, data_samples)
            return raw
        elif mode == 'predict':
            # Pre-process runs in BaseModel.val_step / test_step
            predictions = self.forward_test(inputs, data_samples)
            predictions = self.convert_to_datasample(predictions, data_samples,
                                                     inputs)
            return predictions
        elif mode == 'loss':
            raise NotImplementedError('This mode should not be used in '
                                      'current training schedule. Please use '
                                      '`train_step` for training.')
        else:
            raise ValueError('Invalid forward mode.')

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
            dict: Dict with loss, information for logger, the number of
                samples and results for visualization.
        """
        data = self.data_preprocessor(data, True)
        batch_inputs, data_samples = data['inputs'], data['data_samples']
        log_vars = {}

        masked_img = batch_inputs  # float
        # gt_img: float [-1, 1], mask: uint8 [0/1]
        gt_img, mask = data_samples.gt_img, data_samples.mask
        mask = mask.float()

        # get common output from encdec
        input_x = torch.cat([masked_img, mask], dim=1)
        fake_res = self.generator(input_x)
        fake_img = gt_img * (1. - mask) + fake_res * mask

        # discriminator training step
        if self.train_cfg.disc_step > 0:
            set_requires_grad(self.disc, True)
            disc_losses = self.forward_train_d(
                fake_img.detach(), False, is_disc=True)
            loss_disc, log_vars_d = self.parse_losses(disc_losses)
            log_vars.update(log_vars_d)
            optim_wrapper['disc'].zero_grad()
            loss_disc.backward()

            disc_losses = self.forward_train_d(gt_img, True, is_disc=True)
            loss_disc, log_vars_d = self.parse_losses(disc_losses)
            log_vars.update(log_vars_d)
            loss_disc.backward()

            if self.with_gp_loss:
                loss_d_gp = self.loss_gp(
                    self.disc, gt_img, fake_img, mask=mask)
                loss_disc, log_vars_d = self.parse_losses(
                    dict(loss_gp=loss_d_gp))
                log_vars.update(log_vars_d)
                loss_disc.backward()

            optim_wrapper['disc'].step()

            self.disc_step_count = (self.disc_step_count +
                                    1) % self.train_cfg.disc_step
            if self.disc_step_count != 0:
                # results contain the data for visualization
                results = dict(
                    gt_img=gt_img.cpu(),
                    masked_img=masked_img.cpu(),
                    fake_res=fake_res.cpu(),
                    fake_img=fake_img.cpu())

                # outputs = dict(
                #     log_vars=log_vars,
                #     num_samples=len(data_batch['gt_img'].data),
                #     results=results)

                return log_vars

        # generator (encdec) training step, results contain the data
        # for visualization
        if self.with_gan:
            set_requires_grad(self.disc, False)
        results, g_losses = self.generator_loss(fake_res, fake_img, gt_img,
                                                mask, masked_img)
        loss_g, log_vars_g = self.parse_losses(g_losses)
        log_vars.update(log_vars_g)
        optim_wrapper['generator'].zero_grad()
        loss_g.backward()
        optim_wrapper['generator'].step()

        # outputs = dict(
        #     log_vars=log_vars,
        #     num_samples=len(data_batch['gt_img'].data),
        #     results=results)

        return log_vars

    def forward_train(self, *args, **kwargs) -> None:
        """Forward function for training.

        In this version, we do not use this interface.
        """
        raise NotImplementedError('This interface should not be used in '
                                  'current training schedule. Please use '
                                  '`train_step` for training.')

    def forward_train_d(self, data_batch: torch.Tensor, is_real: bool,
                        is_disc: bool) -> dict:
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

    def generator_loss(self, fake_res: torch.Tensor, fake_img: torch.Tensor,
                       gt: torch.Tensor, mask: torch.Tensor,
                       masked_img: torch.Tensor) -> Tuple[dict, dict]:
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
            tuple(dict): Dict contains the results computed within this \
                function for visualization and dict contains the loss items \
                computed in this function.
        """
        loss = dict()

        if self.with_gan:
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

    def forward_tensor(self, inputs: torch.Tensor, data_samples: SampleList
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward function in tensor mode.

        Args:
            inputs (torch.Tensor): Input tensor.
            data_samples (List[dict]): List of data sample dict.

        Returns:
            tuple: Direct output of the generator and composition of `fake_res`
                and ground-truth image.
        """
        # Pre-process runs in BaseModel.val_step / test_step
        masked_imgs = inputs  # N,3,H,W

        masks = data_samples.mask  # N,1,H,W
        input_xs = torch.cat([masked_imgs, masks], dim=1)  # N,4,H,W
        fake_reses = self.generator(input_xs)
        fake_imgs = fake_reses * masks + masked_imgs * (1. - masks)
        return fake_reses, fake_imgs

    def forward_test(self, inputs: torch.Tensor,
                     data_samples: SampleList) -> DataSample:
        """Forward function for testing.

        Args:
            inputs (torch.Tensor): Input tensor.
            data_samples (List[dict]): List of data sample dict.

        Returns:
            predictions (List[DataSample]): List of prediction saved in
                DataSample.
        """
        fake_reses, fake_imgs = self.forward_tensor(inputs, data_samples)

        predictions = []
        fake_reses = self.data_preprocessor.destruct(fake_reses, data_samples)
        fake_imgs = self.data_preprocessor.destruct(fake_imgs, data_samples)

        # create a stacked data sample here
        predictions = DataSample(
            fake_res=fake_reses, fake_img=fake_imgs, pred_img=fake_imgs)

        return predictions

    def convert_to_datasample(self, predictions: DataSample,
                              data_samples: DataSample,
                              inputs: Optional[torch.Tensor]
                              ) -> List[DataSample]:
        """Add predictions and destructed inputs (if passed) to data samples.

        Args:
            predictions (DataSample): The predictions of the model.
            data_samples (DataSample): The data samples loaded from
                dataloader.
            inputs (Optional[torch.Tensor]): The input of model. Defaults to
                None.

        Returns:
            List[DataSample]: Modified data samples.
        """
        if inputs is not None:
            destructed_input = self.data_preprocessor.destruct(
                inputs, data_samples, 'img')
            data_samples.set_tensor_data({'input': destructed_input})
        data_samples = data_samples.split()
        predictions = predictions.split()

        for data_sample, pred in zip(data_samples, predictions):
            data_sample.output = pred

        return data_samples

    def forward_dummy(self, x: torch.Tensor) -> torch.Tensor:
        """Forward dummy function for getting flops.

        Args:
            x (torch.Tensor): Input tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Results tensor with shape of (n, 3, h, w).
        """
        res = self.generator(x)

        return res
