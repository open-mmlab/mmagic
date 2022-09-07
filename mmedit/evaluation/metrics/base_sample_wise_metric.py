# Copyright (c) OpenMMLab. All rights reserved.
"""Evaluation metrics based on each sample."""

from typing import List, Optional, Sequence

from mmengine.evaluator import BaseMetric

from mmedit.registry import METRICS
from .metrics_utils import average, obtain_data


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
        device (str): Device used to place torch tensors to compute metrics.
            Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Default: None
        scaling (float, optional): Scaling factor for final metric.
            E.g. scaling=100 means the final metric will be amplified by 100
            for output. Default: 1
    """

    metric = None

    def __init__(self,
                 gt_key: str = 'gt_img',
                 pred_key: str = 'pred_img',
                 mask_key: Optional[str] = None,
                 scaling=1,
                 device='cpu',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device, prefix)

        self.gt_key = gt_key
        self.pred_key = pred_key
        self.mask_key = mask_key
        self.scaling = scaling
        self.device = device

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (List): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """

        result = average(results, self.metric) * self.scaling

        return {self.metric: result}

    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]) -> None:
        """Process one batch of data and predictions.

        Args:
            data_batch (Sequence[dict]): A batch of data
                from the dataloader.
            predictions (Sequence[dict]): A batch of outputs from
                the model.
        """

        for data in data_samples:
            prediction = data['output']

            self.channel_order = 'rgb'
            metainfo = data
            if 'gt_channel_order' in metainfo:
                self.channel_order = metainfo['gt_channel_order']
            elif 'img_channel_order' in metainfo:
                self.channel_order = metainfo['img_channel_order']

            gt = obtain_data(data, self.gt_key, self.device)
            pred = obtain_data(prediction, self.pred_key, self.device)
            if self.mask_key is not None:
                mask = obtain_data(data, self.mask_key)
                mask[mask != 0] = 1
            else:
                mask = 1 - pred * 0

            if len(gt.shape) <= 3:
                result = self.process_image(gt, pred, mask)
            else:
                result_sum = 0
                for i in range(gt.shape[0]):
                    result_sum += self.process_image(gt[i], pred[i], mask[i])
                result = result_sum / gt.shape[0]

            self.results.append({self.metric: result})

    def process_image(self, gt, pred, mask):
        return 0
