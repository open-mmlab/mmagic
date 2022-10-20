# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence

from mmeval import SNR as SNR_MMEVAL

from mmedit.registry import METRICS
from .metrics_utils import obtain_data


@METRICS.register_module()
class SNR(SNR_MMEVAL):
    """Signal-to-Noise Ratio. A wrapper of :class:`mmeval.SNR`.

    Ref: https://en.wikipedia.org/wiki/Signal-to-noise_ratio

    Args:
        gt_key (str): Key of ground-truth. Default: 'gt_img'
        pred_key (str): Key of prediction. Default: 'pred_img'
        crop_border (int): Cropped pixels in each edges of an image. These
            pixels are not involved in the SNR calculation. Default: 0.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'CHW'.
        convert_to (str): Whether to convert the images to other color models.
            If None, the images are not altered. When computing for 'Y',
            the images are assumed to be in BGR order. Options are 'Y' and
            None. Default: None.
        channel_order (str): The channel order of image. Default: 'rgb'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Default: None
        dist_backend (str | None): The name of the distributed communication
            backend. Refer to :class:`mmeval.BaseMetric`.
            Defaults to 'torch_cuda'.

    Metrics:
        - SNR (float): Signal-to-Noise Ratio
    """

    metric = 'SNR'

    def __init__(self,
                 gt_key: str = 'gt_img',
                 pred_key: str = 'pred_img',
                 input_order='CHW',
                 crop_border=0,
                 convert_to=None,
                 channel_order: str = 'rgb',
                 prefix: Optional[str] = None,
                 scaling: float = 1,
                 dist_backend: str = 'torch_cuda',
                 **kwargs) -> None:
        super().__init__(
            crop_border,
            input_order,
            convert_to,
            channel_order,
            dist_backend=dist_backend,
            **kwargs)

        self.gt_key = gt_key
        self.pred_key = pred_key
        self.prefix = prefix
        self.scaling = scaling

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

            metainfo = data
            if 'gt_channel_order' in metainfo:
                channel_order = metainfo['gt_channel_order']
            elif 'img_channel_order' in metainfo:
                channel_order = metainfo['img_channel_order']
            else:
                channel_order = self.channel_order

            # convert to list of np.ndarray
            gt = [obtain_data(data, self.gt_key).numpy()]
            pred = [obtain_data(prediction, self.pred_key).numpy()]

            self.add(pred, gt, channel_order)

    def evaluate(self, *args, **kwargs):
        """Returns metric results and print pretty table of metrics per class.

        This method would be invoked by ``mmengine.Evaluator``.
        """
        # print('snr: ', *args, **kwargs)
        metric_results = self.compute(*args, **kwargs)
        self.reset()

        key_template = f'{self.prefix}/{{}}' if self.prefix else '{}'
        return {
            key_template.format(k): v * self.scaling
            for k, v in metric_results.items()
        }
