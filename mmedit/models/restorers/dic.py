import numbers
import os.path as osp
from collections import OrderedDict

import mmcv
import torch

from mmedit.core import tensor2img
from mmedit.models.common import ImgNormalize
from ..builder import build_backbone, build_loss
from ..registry import MODELS
from .basic_restorer import BasicRestorer


@MODELS.register_module()
class DIC(BasicRestorer):
    """DIC model for Face Super-Resolution.

    Paper: Deep Face Super-Resolution with Iterative Collaboration between
        Attentive Recovery and Landmark Estimation.

    Args:
        generator (dict): Config for the generator.
        pixel_loss (dict): Config for the pixel loss.
        align_loss (dict): Config for thr align loss.
        train_cfg (dict): Config for train. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path for pretrained model. Default: None.
    """

    def __init__(self,
                 generator,
                 pixel_loss,
                 align_loss,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(BasicRestorer, self).__init__()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # model
        self.generator = build_backbone(generator)
        self.img_denormalize = ImgNormalize(
            pixel_range=1,
            img_mean=(0.509, 0.424, 0.378),
            img_std=(1., 1., 1.),
            sign=1)

        # loss
        self.pixel_loss = build_loss(pixel_loss)
        self.align_loss = build_loss(align_loss)

        # pretrained
        self.init_weights(pretrained)

    def forward(self, lq, gt=None, test_mode=False, **kwargs):
        """Forward function.

        Args:
            lq (Tensor): Input lq images.
            gt (Tensor): Ground-truth image. Default: None.
            test_mode (bool): Whether in test mode or not. Default: False.
            kwargs (dict): Other arguments.
        """

        if test_mode:
            return self.forward_test(lq, gt=gt, **kwargs)

        return self.generator.forward(lq)

    def train_step(self, data_batch, optimizer):
        """Train step.

        Args:
            data_batch (dict): A batch of data, which requires
                'lq', 'gt'
            optimizer (obj): Optimizer.

        Returns:
            dict: Returned output, which includes:
                log_vars, num_samples, results (lq, gt and pred).

        """
        # data
        lq = data_batch['lq']
        gt = data_batch['gt']
        gt_heatmap = data_batch['heatmap']

        # generate
        sr_list, heatmap_list = self(**data_batch, test_mode=False)

        # loss
        losses = OrderedDict()

        loss_pix = 0.0
        loss_align = 0.0
        for step, (sr, heatmap) in enumerate(zip(sr_list, heatmap_list)):
            losses[f'loss_pixel_v{step}'] = self.pixel_loss(sr, gt)
            loss_pix += losses[f'loss_pixel_v{step}']
            losses[f'loss_align_v{step}'] = self.pixel_loss(
                heatmap, gt_heatmap)
            loss_align += losses[f'loss_align_v{step}']

        # parse loss
        loss, log_vars = self.parse_losses(losses)

        # optimize
        optimizer['generator'].zero_grad()
        loss.backward()
        optimizer['generator'].step()

        log_vars.pop('loss')  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars,
            num_samples=len(gt.data),
            results=dict(lq=lq.cpu(), gt=gt.cpu(), output=sr_list[-1].cpu()))

        return outputs

    def forward_test(self,
                     lq,
                     gt=None,
                     meta=None,
                     save_image=False,
                     save_path=None,
                     iteration=None):
        """Testing forward function.

        Args:
            lq (Tensor): LQ image.
            gt (Tensor): GT image.
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

        # generator
        with torch.no_grad():
            sr_list, _ = self.generator.forward(lq)
            pred = sr_list[-1]
            pred = self.img_denormalize(pred)

            if gt is not None:
                gt = self.img_denormalize(gt)

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
            if 'gt_path' in meta[0]:
                pred_path = meta[0]['gt_path']
            else:
                pred_path = meta[0]['lq_path']
            folder_name = osp.splitext(osp.basename(pred_path))[0]
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
