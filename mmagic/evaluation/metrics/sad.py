# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Sequence

import numpy as np
import torch.nn as nn
from mmengine.model import is_model_wrapper
from torch.utils.data.dataloader import DataLoader

from mmagic.registry import METRICS
from .base_sample_wise_metric import BaseSampleWiseMetric
from .metrics_utils import _fetch_data_and_check, average


@METRICS.register_module()
class SAD(BaseSampleWiseMetric):
    """Sum of Absolute Differences metric for image matting.

    This metric compute per-pixel absolute difference and sum across all
    pixels.
    i.e. sum(abs(a-b)) / norm_const

    .. note::

        Current implementation assume image / alpha / trimap array in numpy
        format and with pixel value ranging from 0 to 255.

    .. note::

        pred_alpha should be masked by trimap before passing
        into this metric

    Default prefix: ''

    Args:
        norm_const (int): Divide the result to reduce its magnitude.
            Default to 1000.

    Metrics:
        - SAD (float): Sum of Absolute Differences
    """

    default_prefix = ''
    metric = 'SAD'

    def __init__(
        self,
        norm_const=1000,
        **kwargs,
    ) -> None:
        self.norm_const = norm_const
        super().__init__(**kwargs)

    def prepare(self, module: nn.Module, dataloader: DataLoader):
        self.size = len(dataloader.dataset)
        if is_model_wrapper(module):
            module = module.module
        self.data_preprocessor = module.data_preprocessor

    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]) -> None:
        """Process one batch of data and predictions.

        Args:
            data_batch (Sequence[Tuple[Any, dict]]): A batch of data
                from the dataloader.
            predictions (Sequence[dict]): A batch of outputs from
                the model.
        """
        for data_sample in data_samples:
            pred_alpha, gt_alpha, _ = _fetch_data_and_check(data_sample)

            # divide by 1000 to reduce the magnitude of the result
            sad_sum = np.abs(pred_alpha - gt_alpha).sum() / self.norm_const

            result = {'sad': sad_sum}

            self.results.append(result)

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (dict): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """

        sad = average(results, 'sad')

        return {'SAD': sad}
