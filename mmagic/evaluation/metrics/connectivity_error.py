# Copyright (c) OpenMMLab. All rights reserved.
"""Evaluation metrics used in Image Matting."""

from typing import List, Sequence

import cv2
import numpy as np
import torch.nn as nn
from mmengine.model import is_model_wrapper
from torch.utils.data.dataloader import DataLoader

from mmagic.registry import METRICS
from .base_sample_wise_metric import BaseSampleWiseMetric
from .metrics_utils import _fetch_data_and_check, average


@METRICS.register_module()
class ConnectivityError(BaseSampleWiseMetric):
    """Connectivity error for evaluating alpha matte prediction.

    .. note::

        Current implementation assume image / alpha / trimap array in numpy
        format and with pixel value ranging from 0 to 255.

    .. note::

        pred_alpha should be masked by trimap before passing
        into this metric

    Args:
        step (float): Step of threshold when computing intersection between
            `alpha` and `pred_alpha`. Default to 0.1 .
        norm_const (int): Divide the result to reduce its magnitude.
            Default to 1000.

    Default prefix: ''

    Metrics:
        - ConnectivityError (float): Connectivity Error
    """

    metric = 'ConnectivityError'

    def __init__(
        self,
        step=0.1,
        norm_constant=1000,
        **kwargs,
    ) -> None:
        self.step = step
        self.norm_constant = norm_constant
        super().__init__(**kwargs)

    def prepare(self, module: nn.Module, dataloader: DataLoader):
        self.size = len(dataloader.dataset)
        if is_model_wrapper(module):
            module = module.module
        self.data_preprocessor = module.data_preprocessor

    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[dict]): A batch of data from the dataloader.
            predictions (Sequence[dict]): A batch of outputs from
                the model.
        """

        for data_sample in data_samples:
            pred_alpha, gt_alpha, trimap = _fetch_data_and_check(data_sample)

            thresh_steps = np.arange(0, 1 + self.step, self.step)
            round_down_map = -np.ones_like(gt_alpha)
            for i in range(1, len(thresh_steps)):
                gt_alpha_thresh = gt_alpha >= thresh_steps[i]
                pred_alpha_thresh = pred_alpha >= thresh_steps[i]
                intersection = gt_alpha_thresh & pred_alpha_thresh
                intersection = intersection.astype(np.uint8)

                # connected components
                _, output, stats, _ = cv2.connectedComponentsWithStats(
                    intersection, connectivity=4)
                # start from 1 in dim 0 to exclude background
                size = stats[1:, -1]

                # largest connected component of the intersection
                omega = np.zeros_like(gt_alpha)
                if len(size) != 0:
                    max_id = np.argmax(size)
                    # plus one to include background
                    omega[output == max_id + 1] = 1

                mask = (round_down_map == -1) & (omega == 0)
                round_down_map[mask] = thresh_steps[i - 1]
            round_down_map[round_down_map == -1] = 1

            gt_alpha_diff = gt_alpha - round_down_map
            pred_alpha_diff = pred_alpha - round_down_map
            # only calculate difference larger than or equal to 0.15
            gt_alpha_phi = 1 - gt_alpha_diff * (gt_alpha_diff >= 0.15)
            pred_alpha_phi = 1 - pred_alpha_diff * (pred_alpha_diff >= 0.15)

            connectivity_error = np.sum(
                np.abs(gt_alpha_phi - pred_alpha_phi) * (trimap == 128))

            # divide by 1000 to reduce the magnitude of the result
            connectivity_error /= self.norm_constant

            self.results.append({'conn_err': connectivity_error})

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (dict): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """

        conn_err = average(results, 'conn_err')

        return {'ConnectivityError': conn_err}
