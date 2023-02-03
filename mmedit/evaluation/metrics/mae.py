# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence

from mmeval import MAE as MAE_MMEVAL

from mmedit.registry import METRICS
from .metrics_utils import obtain_data


@METRICS.register_module()
class MAE(MAE_MMEVAL):
    """Mean Absolute Error metric for image.

    mean(abs(a-b))

    Args:
        gt_key (str): Key of ground-truth. Default: 'gt_img'
        pred_key (str): Key of prediction. Default: 'pred_img'
        mask_key (str, optional): Key of mask, if mask_key is None, calculate
            all regions. Default: None
        scaling (float, optional): Scaling factor for final metric.
            E.g. scaling=100 means the final metric will be amplified by 100
            for output. Default: 1
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Default: None.
        dist_backend (str | None): The name of the distributed communication
            backend. Refer to :class:`mmeval.BaseMetric`.
            Defaults to 'torch_cuda'.

    Metrics:
        - MAE (float): Mean of Absolute Error
    """

    metric = 'MAE'

    def __init__(self,
                 gt_key: str = 'gt_img',
                 pred_key: str = 'pred_img',
                 mask_key: Optional[str] = None,
                 scaling: float = 1,
                 prefix: Optional[str] = None,
                 dist_backend: str = 'torch_cuda',
                 **kwargs) -> None:
        super().__init__(dist_backend=dist_backend, **kwargs)

        self.gt_key = gt_key
        self.pred_key = pred_key
        self.mask_key = mask_key
        self.scaling = scaling
        self.prefix = prefix

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

            # convert to list of np.ndarray
            gt = [obtain_data(data, self.gt_key).numpy()]
            pred = [obtain_data(prediction, self.pred_key).numpy()]
            if self.mask_key is not None:
                mask = obtain_data(data, self.mask_key).numpy()
                mask[mask != 0] = 1
                mask = [mask]
            else:
                mask = [1 - p * 0 for p in pred]

            self.add(pred, gt, mask)

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
