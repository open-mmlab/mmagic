# Copyright (c) OpenMMLab. All rights reserved.
import numbers
import os.path as osp

import mmcv

from mmedit.core import tensor2img
from ..builder import build_loss
from ..registry import MODELS
from .basic_restorer import BasicRestorer


@MODELS.register_module()
class TDAN(BasicRestorer):
    """TDAN model for video super-resolution.

    Paper:
        TDAN: Temporally-Deformable Alignment Network for Video Super-
        Resolution, CVPR, 2020

    Args:
        generator (dict): Config for the generator structure.
        pixel_loss (dict): Config for pixel-wise loss.
        lq_pixel_loss (dict): Config for pixel-wise loss for the LQ images.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path for pretrained model. Default: None.
    """

    def __init__(self,
                 generator,
                 pixel_loss,
                 lq_pixel_loss,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__(generator, pixel_loss, train_cfg, test_cfg,
                         pretrained)

        self.lq_pixel_loss = build_loss(lq_pixel_loss)

    def forward_train(self, lq, gt):
        """Training forward function.

        Args:
            lq (Tensor): LQ Tensor with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w).

        Returns:
            Dict: Output dictionary containing necessary information.
        """
        losses = dict()
        output, aligned_lqs = self.generator(lq)

        # loss on the HR image
        loss_pix = self.pixel_loss(output, gt)
        losses['loss_pix'] = loss_pix
        # loss on the aligned LR images
        t = aligned_lqs.size(1)
        lq_ref = lq[:, t // 2:t // 2 + 1, :, :, :].expand(-1, t, -1, -1, -1)
        loss_pix_lq = self.lq_pixel_loss(aligned_lqs, lq_ref)
        losses['loss_pix_lq'] = loss_pix_lq

        outputs = dict(
            losses=losses,
            num_samples=len(gt.data),
            results=dict(lq=lq.cpu(), gt=gt.cpu(), output=output.cpu()))

        return outputs

    def evaluate(self, output, gt):
        """Evaluation function.

        Args:
            output (Tensor): Model output with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w).

        Returns:
            dict: Evaluation results.
        """
        crop_border = self.test_cfg.crop_border
        convert_to = self.test_cfg.get('convert_to', None)

        output = tensor2img(output)
        gt = tensor2img(gt)

        eval_result = dict()
        for metric in self.test_cfg.metrics:
            eval_result[metric] = self.allowed_metrics[metric](
                output, gt, crop_border, convert_to=convert_to)
        return eval_result

    def forward_test(self,
                     lq,
                     gt=None,
                     meta=None,
                     save_image=False,
                     save_path=None,
                     iteration=None):
        """Testing forward function.

        Args:
            lq (Tensor): LQ Tensor with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w). Default: None.
            save_image (bool): Whether to save image. Default: False.
            save_path (str): Path to save image. Default: None.
            iteration (int): Iteration for the saving image name.
                Default: None.

        Returns:
            dict: Output results.
        """
        output = self.generator(lq)[0]  # keep only the HR output

        # normalize from [-0.5, 0.5] to [0, 1] (following official)
        output += 0.5

        if self.test_cfg is not None and self.test_cfg.get('metrics', None):
            assert gt is not None, (
                'evaluation with metrics must have gt images.')
            gt += 0.5  # normalize from [-0.5, 0.5] to [0, 1]
            results = dict(eval_result=self.evaluate(output, gt))
        else:
            results = dict(lq=lq.cpu(), output=output.cpu())
            if gt is not None:
                results['gt'] = gt.cpu()

        # save image
        if save_image:
            gt_path = meta[0]['gt_path'][0]
            clip_name = meta[0]['key'].split('/')[0]
            frame_name = osp.splitext(osp.basename(gt_path))[0]

            if isinstance(iteration, numbers.Number):
                save_path = osp.join(save_path, clip_name,
                                     f'{frame_name}-{iteration + 1:06d}.png')
            elif iteration is None:
                save_path = osp.join(save_path, clip_name, f'{frame_name}.png')
            else:
                raise ValueError('iteration should be number or None, '
                                 f'but got {type(iteration)}')
            mmcv.imwrite(tensor2img(output), save_path)

        return results
