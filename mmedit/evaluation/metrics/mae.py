# Copyright (c) OpenMMLab. All rights reserved.
"""Evaluation metrics based on pixels."""

import numpy as np
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from mmedit.registry import METRICS
from .base_sample_wise_metric import BaseSampleWiseMetric


@METRICS.register_module()
class MAE(BaseSampleWiseMetric):
    """Mean Absolute Error metric for image.

    mean(abs(a-b))

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
        - MAE (float): Mean of Absolute Error
    """

    metric = 'MAE'

    def prepare(self, module: nn.Module, dataloader: DataLoader):
        self.SAMPLER_MODE = 'normal'

    def get_metric_sampler(self, model: nn.Module, dataloader: DataLoader,
                           metrics) -> DataLoader:
        """Get sampler for normal metrics. Directly returns the dataloader.

        Args:
            model (nn.Module): Model to evaluate.
            dataloader (DataLoader): Dataloader for real images.
            metrics (List['GenMetric']): Metrics with the same sample mode.

        Returns:
            DataLoader: Default sampler for normal metrics.
        """
        return dataloader

    def evaluate(self, size=None) -> dict:
        if size is None:
            size = len(self.results)
        return super().evaluate(size)

    def process_image(self, gt, pred, mask):
        """Process an image.

        Args:
            gt (Tensor | np.ndarray): GT image.
            pred (Tensor | np.ndarray): Pred image.
            mask (Tensor | np.ndarray): Mask of evaluation.
        Returns:
            result (np.ndarray): MAE result.
        """

        gt = gt / 255.
        pred = pred / 255.

        diff = gt - pred
        diff = abs(diff)

        if self.mask_key is not None:
            diff *= mask  # broadcast for channel dimension
            scale = np.prod(diff.shape) / np.prod(mask.shape)
            result = diff.sum() / (mask.sum() * scale + 1e-12)
        else:
            result = diff.mean()

        return result
