# Copyright (c) OpenMMLab. All rights reserved.
"""Evaluation metrics based on each sample"""

from typing import List, Optional

from mmengine.evaluator import BaseMetric

from mmedit.registry import METRICS
from .utils import average


@METRICS.register_module()
class BaseSampleWiseMetric(BaseMetric):
    """Base sample wise metric of edit.

    Subclass must provide process function.

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
    """

    default_prefix = 'BaseEditMetric'

    def __init__(self,
                 gt_key: str = 'gt_img',
                 pred_key: str = 'pred_img',
                 mask_key: Optional[str] = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device, prefix)

        self.gt_key = gt_key
        self.pred_key = pred_key
        self.mask_key = mask_key

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (dict): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """

        result = average(results, self.prefix)

        return {self.prefix: result}
