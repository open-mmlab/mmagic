# Copyright (c) OpenMMLab. All rights reserved.
import numbers
import os.path as osp

import mmcv

from mmedit.core import tensor2img
from ..registry import MODELS
from .srgan import SRGAN


@MODELS.register_module()
class GLEAN(SRGAN):
    """GLEAN model for single image super-resolution.

    This model is identical to SRGAN except that the output images are
    transformed from [-1, 1] to [0, 1].

    Paper:
    GLEAN: Generative Latent Bank for Large-Factor Image Super-Resolution.
    CVPR, 2021.
    """

    def init_weights(self, pretrained=None):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
        """
        self.generator.init_weights(pretrained=pretrained)

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
        output = self.generator(lq)

        # normalize from [-1, 1] to [0, 1]
        output = (output + 1) / 2.0

        if self.test_cfg is not None and self.test_cfg.get('metrics', None):
            assert gt is not None, (
                'evaluation with metrics must have gt images.')
            gt = (gt + 1) / 2.0  # normalize from [-1, 1] to [0, 1]
            results = dict(eval_result=self.evaluate(output, gt))
        else:
            results = dict(lq=lq.cpu(), output=output.cpu())
            if gt is not None:
                results['gt'] = gt.cpu()

        # save image
        if save_image:
            lq_path = meta[0]['lq_path']
            folder_name = osp.splitext(osp.basename(lq_path))[0]
            if isinstance(iteration, numbers.Number):
                save_path = osp.join(save_path, folder_name,
                                     f'{folder_name}-{iteration + 1:06d}.png')
            elif iteration is None:
                save_path = osp.join(save_path, f'{folder_name}.png')
            else:
                raise ValueError('iteration should be number or None, '
                                 f'but got {type(iteration)}')
            mmcv.imwrite(tensor2img(output), save_path)

        return results
