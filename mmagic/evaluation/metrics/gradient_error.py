# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Sequence

import cv2
import numpy as np
import torch.nn as nn
from mmengine.model import is_model_wrapper
from torch.utils.data.dataloader import DataLoader

from mmagic.registry import METRICS
from ..functional import gauss_gradient
from .base_sample_wise_metric import BaseSampleWiseMetric
from .metrics_utils import _fetch_data_and_check, average


@METRICS.register_module()
class GradientError(BaseSampleWiseMetric):
    """Gradient error for evaluating alpha matte prediction.

    .. note::

        Current implementation assume image / alpha / trimap array in numpy
        format and with pixel value ranging from 0 to 255.

    .. note::

        pred_alpha should be masked by trimap before passing
        into this metric

    Args:
        sigma (float): Standard deviation of the gaussian kernel.
            Defaults to 1.4 .
        norm_const (int): Divide the result to reduce its magnitude.
            Defaults to 1000 .

    Default prefix: ''

    Metrics:
        - GradientError (float): Gradient Error
    """

    metric = 'GradientError'

    def __init__(
        self,
        sigma=1.4,
        norm_constant=1000,
        **kwargs,
    ) -> None:
        self.sigma = sigma
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

            gt_alpha_normed = np.zeros_like(gt_alpha)
            pred_alpha_normed = np.zeros_like(pred_alpha)

            cv2.normalize(gt_alpha, gt_alpha_normed, 1.0, 0.0, cv2.NORM_MINMAX)
            cv2.normalize(pred_alpha, pred_alpha_normed, 1.0, 0.0,
                          cv2.NORM_MINMAX)

            gt_alpha_grad = gauss_gradient(gt_alpha_normed, self.sigma)
            pred_alpha_grad = gauss_gradient(pred_alpha_normed, self.sigma)
            # this is the sum over n samples
            grad_loss = ((gt_alpha_grad - pred_alpha_grad)**2 *
                         (trimap == 128)).sum()

            # divide by 1000 to reduce the magnitude of the result
            grad_loss /= self.norm_constant

            self.results.append({'grad_err': grad_loss})

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (dict): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """

        grad_err = average(results, 'grad_err')

        return {'GradientError': grad_err}
