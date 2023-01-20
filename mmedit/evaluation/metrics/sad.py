# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence

from mmeval import SAD as _SAD

from mmedit.registry import METRICS
from .metrics_utils import _fetch_data_and_check


@METRICS.register_module()
class SAD(_SAD):
    """Sum of Absolute Differences metric for image matting.A wrapper of
    :class:`mmeval.SAD`.

    This metric compute per-pixel absolute difference and sum across all
    pixels.
    i.e. sum(abs(a-b)) / norm_const

    .. note::

        Current implementation assume image / alpha / trimap array in numpy
        format and with pixel value ranging from 0 to 255.

    .. note::

        pred_alpha should be masked by trimap before passing
        into this metric

    .. note::

        norm_const (int): Divide the result to reduce its magnitude.
        Default to 1000.

    Metrics:
        - SAD (float): Sum of Absolute Differences
    """

    def __init__(
        self,
        scaling: float = 1,
        prefix: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.scaling = scaling
        self.prefix = prefix

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
