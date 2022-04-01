# Copyright (c) OpenMMLab. All rights reserved.
import numbers
import os.path as osp

import mmcv
import numpy as np
import torch

from mmedit.core import tensor2img
from ..registry import MODELS
from .basic_restorer import BasicRestorer


@MODELS.register_module()
class BasicVSR(BasicRestorer):
    """BasicVSR model for video super-resolution.

    Note that this model is used for IconVSR.

    Paper:
        BasicVSR: The Search for Essential Components in Video Super-Resolution
        and Beyond, CVPR, 2021

    Args:
        generator (dict): Config for the generator structure.
        pixel_loss (dict): Config for pixel-wise loss.
        ensemble (dict): Config for ensemble. Default: None.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path for pretrained model. Default: None.
    """

    def __init__(self,
                 generator,
                 pixel_loss,
                 ensemble=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__(generator, pixel_loss, train_cfg, test_cfg,
                         pretrained)

        # fix pre-trained networks
        self.fix_iter = train_cfg.get('fix_iter', 0) if train_cfg else 0
        self.is_weight_fixed = False

        # count training steps
        self.register_buffer('step_counter', torch.zeros(1))

        # ensemble
        self.forward_ensemble = None
        if ensemble is not None:
            if ensemble['type'] == 'SpatialTemporalEnsemble':
                from mmedit.models.common.ensemble import \
                    SpatialTemporalEnsemble
                is_temporal = ensemble.get('is_temporal_ensemble', False)
                self.forward_ensemble = SpatialTemporalEnsemble(is_temporal)
            else:
                raise NotImplementedError(
                    'Currently support only '
                    '"SpatialTemporalEnsemble", but got type '
                    f'[{ensemble["type"]}]')

    def check_if_mirror_extended(self, lrs):
        """Check whether the input is a mirror-extended sequence.

        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the
        (t-1-i)-th frame.

        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)
        """

        is_mirror_extended = False
        if lrs.size(1) % 2 == 0:
            lrs_1, lrs_2 = torch.chunk(lrs, 2, dim=1)
            if torch.norm(lrs_1 - lrs_2.flip(1)) == 0:
                is_mirror_extended = True

        return is_mirror_extended

    def train_step(self, data_batch, optimizer):
        """Train step.

        Args:
            data_batch (dict): A batch of data.
            optimizer (obj): Optimizer.

        Returns:
            dict: Returned output.
        """
        # fix SPyNet and EDVR at the beginning
        if self.step_counter < self.fix_iter:
            if not self.is_weight_fixed:
                self.is_weight_fixed = True
                for k, v in self.generator.named_parameters():
                    if 'spynet' in k or 'edvr' in k:
                        v.requires_grad_(False)
        elif self.step_counter == self.fix_iter:
            # train all the parameters
            self.generator.requires_grad_(True)

        outputs = self(**data_batch, test_mode=False)
        loss, log_vars = self.parse_losses(outputs.pop('losses'))

        # optimize
        optimizer['generator'].zero_grad()
        loss.backward()
        optimizer['generator'].step()

        self.step_counter += 1

        outputs.update({'log_vars': log_vars})
        return outputs

    def evaluate(self, output, gt):
        """Evaluation function.

        If the output contains multiple frames, we compute the metric
        one by one and take an average.

        Args:
            output (Tensor): Model output with shape (n, t, c, h, w).
            gt (Tensor): GT Tensor with shape (n, t, c, h, w).

        Returns:
            dict: Evaluation results.
        """
        crop_border = self.test_cfg.crop_border
        convert_to = self.test_cfg.get('convert_to', None)

        eval_result = dict()
        for metric in self.test_cfg.metrics:
            if output.ndim == 5:  # a sequence: (n, t, c, h, w)
                avg = []
                for i in range(0, output.size(1)):
                    output_i = tensor2img(output[:, i, :, :, :])
                    gt_i = tensor2img(gt[:, i, :, :, :])
                    avg.append(self.allowed_metrics[metric](
                        output_i, gt_i, crop_border, convert_to=convert_to))
                eval_result[metric] = np.mean(avg)
            elif output.ndim == 4:  # an image: (n, c, t, w), for Vimeo-90K-T
                output_img = tensor2img(output)
                gt_img = tensor2img(gt)
                value = self.allowed_metrics[metric](
                    output_img, gt_img, crop_border, convert_to=convert_to)
                eval_result[metric] = value

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
            lq (Tensor): LQ Tensor with shape (n, t, c, h, w).
            gt (Tensor): GT Tensor with shape (n, t, c, h, w). Default: None.
            save_image (bool): Whether to save image. Default: False.
            save_path (str): Path to save image. Default: None.
            iteration (int): Iteration for the saving image name.
                Default: None.

        Returns:
            dict: Output results.
        """
        with torch.no_grad():
            if self.forward_ensemble is not None:
                output = self.forward_ensemble(lq, self.generator)
            else:
                output = self.generator(lq)

        # If the GT is an image (i.e. the center frame), the output sequence is
        # turned to an image.
        if gt is not None and gt.ndim == 4:
            t = output.size(1)
            if self.check_if_mirror_extended(lq):  # with mirror extension
                output = 0.5 * (output[:, t // 4] + output[:, -1 - t // 4])
            else:  # without mirror extension
                output = output[:, t // 2]

        if self.test_cfg is not None and self.test_cfg.get('metrics', None):
            assert gt is not None, (
                'evaluation with metrics must have gt images.')
            results = dict(eval_result=self.evaluate(output, gt))
        else:
            results = dict(lq=lq.cpu(), output=output.cpu())
            if gt is not None:
                results['gt'] = gt.cpu()

        # save image
        if save_image:
            if output.ndim == 4:  # an image, key = 000001/0000 (Vimeo-90K)
                img_name = meta[0]['key'].replace('/', '_')
                if isinstance(iteration, numbers.Number):
                    save_path = osp.join(
                        save_path, f'{img_name}-{iteration + 1:06d}.png')
                elif iteration is None:
                    save_path = osp.join(save_path, f'{img_name}.png')
                else:
                    raise ValueError('iteration should be number or None, '
                                     f'but got {type(iteration)}')
                mmcv.imwrite(tensor2img(output), save_path)
            elif output.ndim == 5:  # a sequence, key = 000
                folder_name = meta[0]['key'].split('/')[0]
                for i in range(0, output.size(1)):
                    if isinstance(iteration, numbers.Number):
                        save_path_i = osp.join(
                            save_path, folder_name,
                            f'{i:08d}-{iteration + 1:06d}.png')
                    elif iteration is None:
                        save_path_i = osp.join(save_path, folder_name,
                                               f'{i:08d}.png')
                    else:
                        raise ValueError('iteration should be number or None, '
                                         f'but got {type(iteration)}')
                    mmcv.imwrite(
                        tensor2img(output[:, i, :, :, :]), save_path_i)

        return results
