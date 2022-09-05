# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch

from mmedit.models.base_models import OneStageInpaintor
from mmedit.registry import MODELS
from mmedit.structures import EditDataSample, PixelData


@MODELS.register_module()
class PConvInpaintor(OneStageInpaintor):
    """Inpaintor for Partial Convolution method.

    This inpaintor is implemented according to the paper: Image inpainting for
    irregular holes using partial convolutions
    """

    def forward_test(self, inputs, data_samples):
        """Forward function for testing.

        Args:
            inputs (torch.Tensor): Input tensor.
            data_samples (List[dict]): List of data sample dict.

        Returns:
            dict: Contain output results and eval metrics (if have).
        """
        fake_reses, fake_imgs = self.forward_tensor(inputs, data_samples)

        predictions = []
        for (fr, fi) in zip(fake_reses, fake_imgs):
            fi = (fi * 127.5 + 127.5)
            fr = (fr * 127.5 + 127.5)
            pred = EditDataSample(
                fake_res=fr, fake_img=fi, pred_img=PixelData(data=fi))
            predictions.append(pred)
        return predictions

    def forward_tensor(self, inputs, data_samples):
        """Forward function in tensor mode.

        Args:
            inputs (torch.Tensor): Input tensor.
            data_sample (dict): Dict contains data sample.

        Returns:
            dict: Dict contains output results.
        """

        masked_img = inputs  # N,3,H,W
        masks = torch.stack(
            list(d.mask.data for d in data_samples), dim=0)  # N,1,H,W
        masks = 1. - masks
        masks = masks.repeat(1, 3, 1, 1)
        fake_reses, _ = self.generator(masked_img, masks)
        fake_imgs = fake_reses * (1. - masks) + masked_img * masks
        return fake_reses, fake_imgs

    def train_step(self, data: List[dict], optim_wrapper):
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
        # print(gt_img.min(), gt_img.max(), gt_img.dtype)
        mask = torch.stack([d.mask.data for d in data_samples])  # uint8, {0,1}
        mask = mask.float()

        mask_input = mask.expand_as(gt_img)
        mask_input = 1. - mask_input

        fake_res, final_mask = self.generator(masked_img, mask_input)
        fake_img = gt_img * (1. - mask) + fake_res * mask

        results, g_losses = self.generator_loss(fake_res, fake_img, gt_img,
                                                mask, masked_img)
        loss_g_, log_vars_g = self.parse_losses(g_losses)
        log_vars.update(log_vars_g)
        optim_wrapper.zero_grad()
        optim_wrapper.backward(loss_g_)
        optim_wrapper.step()

        return log_vars
