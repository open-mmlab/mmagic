# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..registry import MODELS
from .basic_interpolator import BasicInterpolator


@MODELS.register_module()
class CAIN(BasicInterpolator):
    """CAIN model for Video Interpolation.

    Paper: Channel Attention Is All You Need for Video Frame Interpolation
    Ref repo: https://github.com/myungsub/CAIN

    Args:
        generator (dict): Config for the generator structure.
        pixel_loss (dict): Config for pixel-wise loss.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path for pretrained model. Default: None.
    """

    def forward_train(self, inputs, target):
        """Training forward function.

        Args:
            inputs (Tensor): Tensor of inputs frames with shape
                (n, 2, c, h, w).
            target (Tensor): Tensor of target frame with shape (n, c, h, w).

        Returns:
            Tensor: Output tensor.
        """
        losses = dict()
        output = self.generator(inputs, padding_flag=False)
        loss_pix = self.pixel_loss(output, target)
        losses['loss_pix'] = loss_pix
        outputs = dict(
            losses=losses,
            num_samples=len(target.data),
            results=dict(
                inputs=inputs.cpu(), target=target.cpu(), output=output.cpu()))
        return outputs

    def forward_test(self,
                     inputs,
                     target=None,
                     meta=None,
                     save_image=False,
                     save_path=None,
                     iteration=None):
        """Testing forward function.

        Args:
            inputs (Tensor): The input Tensor with shape (n, 2, c, h, w).
            target (Tensor): The target Tensor with shape (n, c, h, w).
            meta (list[dict]): Meta data, such as path of target file.
                Default: None.
            save_image (bool): Whether to save image. Default: False.
            save_path (str): Path to save image. Default: None.
            iteration (int): Iteration for the saving image name.
                Default: None.

        Returns:
            dict: Output results, which contain either key(s)
                1. 'eval_result'.
                2. 'inputs', 'pred'.
                3. 'inputs', 'pred', and 'target'.
        """

        # generator
        with torch.no_grad():
            pred = self.generator(inputs, padding_flag=True)
            pred = pred.clamp(0, 1)

        if self.test_cfg is not None and self.test_cfg.get('metrics', None):
            assert target is not None, (
                'evaluation with metrics must have target images.')
            results = dict(eval_result=self.evaluate(pred, target))
        else:
            results = dict(inputs=inputs.cpu(), output=pred.cpu())
            if target is not None:
                results['target'] = target.cpu()

        # save image
        if save_image:
            self._save_image(meta, iteration, save_path, pred)

        return results
