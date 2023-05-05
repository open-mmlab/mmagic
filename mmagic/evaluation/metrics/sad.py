# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence

import torch.nn as nn
from mmengine.model import is_model_wrapper
from mmeval import SumAbsoluteDifferences as _SumAbsoluteDifferences
from torch.utils.data.dataloader import DataLoader

from mmagic.registry import METRICS
from .metrics_utils import _fetch_data_and_check


@METRICS.register_module()
class SAD(_SumAbsoluteDifferences):
    """Sum of Absolute Differences metric for image matting.

    This metric compute per-pixel absolute difference and sum across all
    pixels.
    i.e. sum(abs(a-b)) / norm_const

    .. note::
        norm_const (int): Divide the result to reduce its magnitude.
            Default to 1000.

    .. note::

        Current implementation assume image / alpha / trimap array in numpy
        format and with pixel value ranging from 0 to 255.

    .. note::

        pred_alpha should be masked by trimap before passing
        into this metric

    Default prefix: ''

    Args:
        scaling (float, optional): Scaling factor for final metric.
            E.g. scaling=100 means the final metric will be amplified by 100
            for output. Default: 1

        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Default: None.

    Metrics:
        - SAD (float): Sum of Absolute Differences
    """

    default_prefix = ''
    metric = 'SAD'
    SAMPLER_MODE = 'normal'
    sample_model = 'orig'

    def __init__(
        self,
        scaling: float = 1,
        prefix: Optional[str] = None,
        **kwargs,
    ) -> None:

        super().__init__(**kwargs)
        self.scaling = scaling
        self.prefix = prefix

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
        pred_alphas, gt_alphas = [], []
        for data_sample in data_samples:
            pred_alpha, gt_alpha, _ = _fetch_data_and_check(data_sample)
            pred_alphas.append(pred_alpha)
            gt_alphas.append(gt_alpha)

        self.add(pred_alphas, gt_alphas)

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

    def evaluate(self, *args, **kwargs):
        """Returns metric results and print pretty table of metrics per class.

        This method would be invoked by ``mmengine.Evaluator``.
        """
        metric_results = self.compute(*args, **kwargs)
        self.reset()

        key_template = f'{self.prefix}/{{}}' if self.prefix else '{}'
        return {
            key_template.format(k): v * self.scaling
            for k, v in metric_results.items()
        }
