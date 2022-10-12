# Copyright (c) OpenMMLab. All rights reserved.
"""Evaluation metrics based on pixels."""

from mmedit.registry import METRICS
from .base_sample_wise_metric import BaseSampleWiseMetric


@METRICS.register_module()
class MSE(BaseSampleWiseMetric):
    """Mean Squared Error metric for image.

    mean((a-b)^2)

    Args:

        gt_key (str): Key of ground-truth. Default: 'gt_img'
        pred_key (str): Key of prediction. Default: 'pred_img'
        mask_key (str, optional): Key of mask, if mask_key is None, calculate
            all regions. Default: None
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Default: None

    Metrics:
        - MSE (float): Mean of Squared Error
    """

    metric = 'MSE'

    def process_image(self, gt, pred, mask):
        """Process an image.

        Args:
            gt (Torch | np.ndarray): GT image.
            pred (Torch | np.ndarray): Pred image.
            mask (Torch | np.ndarray): Mask of evaluation.
        Returns:
            result (np.ndarray): MSE result.
        """

        gt = gt / 255.
        pred = pred / 255.

        diff = gt - pred
        diff *= diff

        if self.mask_key is not None:
            diff *= mask
            result = diff.sum() / mask.sum()
        else:
            result = diff.mean()

        return result
