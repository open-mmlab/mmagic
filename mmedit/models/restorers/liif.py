# Copyright (c) OpenMMLab. All rights reserved.
import math
import numbers
import os.path as osp

import mmcv
import torch

from mmedit.core import tensor2img
from ..registry import MODELS
from .basic_restorer import BasicRestorer


@MODELS.register_module()
class LIIF(BasicRestorer):
    """LIIF model for single image super-resolution.

    Paper: Learning Continuous Image Representation with
           Local Implicit Image Function

    Args:
        generator (dict): Config for the generator.
        pixel_loss (dict): Config for the pixel loss.
        rgb_mean (tuple[float]): Data mean.
            Default: (0.5, 0.5, 0.5).
        rgb_std (tuple[float]): Data std.
            Default: (0.5, 0.5, 0.5).
        train_cfg (dict): Config for train. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path for pretrained model. Default: None.
    """

    def __init__(self,
                 generator,
                 pixel_loss,
                 rgb_mean=(0.5, 0.5, 0.5),
                 rgb_std=(0.5, 0.5, 0.5),
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__(
            generator,
            pixel_loss,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

        # norm
        rgb_mean = torch.FloatTensor(rgb_mean)
        rgb_std = torch.FloatTensor(rgb_std)
        self.lq_mean = rgb_mean.view(1, -1, 1, 1)
        self.lq_std = rgb_std.view(1, -1, 1, 1)
        self.gt_mean = rgb_mean.view(1, 1, -1)
        self.gt_std = rgb_std.view(1, 1, -1)

    def train_step(self, data_batch, optimizer):
        """Train step.

        Args:
            data_batch (dict): A batch of data, which requires
                'coord', 'lq', 'gt', 'cell'
            optimizer (obj): Optimizer.

        Returns:
            dict: Returned output, which includes:
                log_vars, num_samples, results (lq, gt and pred).
        """
        # data
        coord = data_batch['coord']
        cell = data_batch['cell']
        lq = data_batch['lq']
        gt = data_batch['gt']

        # norm
        self.lq_mean = self.lq_mean.to(lq)
        self.lq_std = self.lq_std.to(lq)
        self.gt_mean = self.gt_mean.to(gt)
        self.gt_std = self.gt_std.to(gt)
        lq = (lq - self.lq_mean) / self.lq_std
        gt = (gt - self.gt_mean) / self.gt_std

        # generator
        pred = self.generator(lq, coord, cell)

        # loss
        losses = dict()
        log_vars = dict()
        losses['loss_pix'] = self.pixel_loss(pred, gt)

        # parse loss
        loss, log_vars = self.parse_losses(losses)

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log_vars.pop('loss')  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars,
            num_samples=len(gt.data),
            results=dict(lq=lq.cpu(), gt=gt.cpu(), output=pred.cpu()))

        return outputs

    def forward_test(self,
                     lq,
                     gt,
                     coord,
                     cell,
                     meta=None,
                     save_image=False,
                     save_path=None,
                     iteration=None):
        """Testing forward function.

        Args:
            lq (Tensor): LQ image.
            gt (Tensor): GT image.
            coord (Tensor): Coord tensor.
            cell (Tensor): Cell tensor.
            meta (list[dict]): Meta data, such as path of GT file.
                Default: None.
            save_image (bool): Whether to save image. Default: False.
            save_path (str): Path to save image. Default: None.
            iteration (int): Iteration for the saving image name.
                Default: None.

        Returns:
            dict: Output results, which contain either key(s)
                1. 'eval_result'.
                2. 'lq', 'pred'.
                3. 'lq', 'pred', 'gt'.
        """

        # norm
        self.lq_mean = self.lq_mean.to(lq)
        self.lq_std = self.lq_std.to(lq)
        lq = (lq - self.lq_mean) / self.lq_std

        # generator
        with torch.no_grad():
            pred = self.generator(lq, coord, cell, test_mode=True)
            self.gt_mean = self.gt_mean.to(pred)
            self.gt_std = self.gt_std.to(pred)
            pred = pred * self.gt_std + self.gt_mean
            pred.clamp_(0, 1)

        # reshape for eval
        ih, iw = lq.shape[-2:]
        s = math.sqrt(coord.shape[1] / (ih * iw))
        shape = [lq.shape[0], round(ih * s), round(iw * s), 3]
        pred = pred.view(*shape).permute(0, 3, 1, 2).contiguous()
        if gt is not None:
            gt = gt.view(*shape).permute(0, 3, 1, 2).contiguous()

        if self.test_cfg is not None and self.test_cfg.get('metrics', None):
            assert gt is not None, (
                'evaluation with metrics must have gt images.')
            results = dict(eval_result=self.evaluate(pred, gt))
        else:
            results = dict(lq=lq.cpu(), output=pred.cpu())
            if gt is not None:
                results['gt'] = gt.cpu()

        # save image
        if save_image:
            gt_path = meta[0]['gt_path']
            folder_name = osp.splitext(osp.basename(gt_path))[0]
            if isinstance(iteration, numbers.Number):
                save_path = osp.join(save_path, folder_name,
                                     f'{folder_name}-{iteration + 1:06d}.png')
            elif iteration is None:
                save_path = osp.join(save_path, f'{folder_name}.png')
            else:
                raise ValueError('iteration should be number or None, '
                                 f'but got {type(iteration)}')
            mmcv.imwrite(tensor2img(pred), save_path)

        return results

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """

        self.generator.init_weights(pretrained, strict)
