# Copyright (c) OpenMMLab. All rights reserved.

from typing import Optional, Sequence

from mmeval import NIQE as NIQE_MMEVAL

from mmedit.registry import METRICS
from .metrics_utils import obtain_data


@METRICS.register_module()
class NIQE(NIQE_MMEVAL):
    """Calculate NIQE (Natural Image Quality Evaluator) metric.

    Ref: Making a "Completely Blind" Image Quality Analyzer.
    This implementation could produce almost the same results as the official
    MATLAB codes: http://live.ece.utexas.edu/research/quality/niqe_release.zip

    We use the official params estimated from the pristine dataset.
    We use the recommended block size (96, 96) without overlaps.

    Args:

        key (str): Key of image. Default: 'pred_img'
        is_predicted (bool): If the image is predicted, it will be picked from
            predictions; otherwise, it will be picked from data_batch.
            Default: True
        crop_border (int): Cropped pixels in each edges of an image. These
            pixels are not involved in the NIQE calculation. Default: 0.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'CHW'.
        convert_to (str): Whether to convert the images to other color models.
            If None, the images are not altered. When computing for 'Y',
            the images are assumed to be in BGR order. Options are 'Y' and
            None. Default: 'gray'.
        channel_order (str): The channel order of image. Default: 'rgb'.
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
        - NIQE (float): Natural Image Quality Evaluator
    """

    metric = 'NIQE'

    def __init__(self,
                 key: str = 'pred_img',
                 is_predicted: bool = True,
                 crop_border=0,
                 input_order='CHW',
                 convert_to='gray',
                 channel_order: str = 'rgb',
                 scaling: float = 1,
                 prefix: Optional[str] = None,
                 dist_backend: str = 'torch_cuda',
                 **kwargs) -> None:
        super().__init__(
            crop_border=crop_border,
            input_order=input_order,
            convert_to=convert_to,
            channel_order=channel_order,
            dist_backend=dist_backend,
            **kwargs)

        self.key = key
        self.is_predicted = is_predicted
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
            pred = [obtain_data(prediction, self.pred_key).numpy()]
            self.add(pred)

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
