# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmedit.data_element import EditDataSample, PixelData
from mmedit.registry import MODELS
from .one_stage import OneStageInpaintor


@MODELS.register_module()
class PConvInpaintor(OneStageInpaintor):

    def forward_test(self, inputs, data_samples):
        """Forward function for testing.

        Args:
            masked_img (torch.Tensor): Tensor with shape of (n, 3, h, w).
            mask (torch.Tensor): Tensor with shape of (n, 1, h, w).
            save_image (bool, optional): If True, results will be saved as
                image. Defaults to False.
            save_path (str, optional): If given a valid str, the results will
                be saved in this path. Defaults to None.
            iteration (int, optional): Iteration number. Defaults to None.

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
        masks = masks.tile(1, 3, 1, 1)
        fake_reses, _ = self.generator(masked_img, masks)
        fake_imgs = fake_reses * (1. - masks) + masked_img * masks
        return fake_reses, fake_imgs

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

        mask_input = mask.expand_as(gt_img)
        mask_input = 1. - mask_input

        fake_res, final_mask = self.generator(masked_img, mask_input)
        fake_img = gt_img * (1. - mask) + fake_res * mask

        results, g_losses = self.generator_loss(fake_res, fake_img, data_batch)
        loss_g_, log_vars_g = self.parse_losses(g_losses)
        log_vars.update(log_vars_g)
        optimizer['generator'].zero_grad()
        loss_g_.backward()
        optimizer['generator'].step()

        results.update(dict(final_mask=final_mask))
        outputs = dict(
            log_vars=log_vars,
            num_samples=len(data_batch['gt_img'].data),
            results=results)

        return outputs

    def forward_dummy(self, x):
        mask = x[:, -3:, ...].clone()
        x = x[:, :-3, ...]
        res, _ = self.generator(x, mask)

        return res
