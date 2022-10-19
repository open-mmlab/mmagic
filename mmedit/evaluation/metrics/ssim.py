# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence

from mmeval import SSIM as SSIM_MMEVAL

from mmedit.registry import METRICS
from .metrics_utils import obtain_data


@METRICS.register_module()
class SSIM(SSIM_MMEVAL):
    """Calculate SSIM (structural similarity). A wrapper of
    :class:`mmeval.SSIM`.

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        gt_key (str): Key of ground-truth. Default: 'gt_img'
        pred_key (str): Key of prediction. Default: 'pred_img'
        crop_border (int): Cropped pixels in each edges of an image. These
            pixels are not involved in the PSNR calculation. Default: 0.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        convert_to (str): Whether to convert the images to other color models.
            If None, the images are not altered. When computing for 'Y',
            the images are assumed to be in BGR order. Options are 'Y' and
            None. Default: None.
        channel_order (str): The channel order of image. Default: 'rgb'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Default: None
        scaling (float, optional): Scaling factor for final metric.
            E.g. scaling=100 means the final metric will be amplified by 100
            for output. Default: 1
        **kwargs: Keyword parameters passed to :class:`BaseMetric`.

    Metrics:
        - SSIM (float): Structural similarity
    """

    def __init__(self,
                 gt_key: str = 'gt_img',
                 pred_key: str = 'pred_img',
                 input_order: str = 'CHW',
                 crop_border: int = 0,
                 convert_to: Optional[str] = None,
                 channel_order: str = 'rgb',
                 prefix: Optional[str] = None,
                 scaling: float = 1,
                 **kwargs) -> None:
        super().__init__(input_order, crop_border, convert_to, channel_order,
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

            channel_order = 'rgb'
            metainfo = data
            if 'gt_channel_order' in metainfo:
                channel_order = metainfo['gt_channel_order']
            elif 'img_channel_order' in metainfo:
                channel_order = metainfo['img_channel_order']

            gt = obtain_data(data, self.gt_key)
            pred = obtain_data(prediction, self.pred_key)

            gt = [sample.numpy() for sample in gt]
            pred = [sample.numpy() for sample in pred]

            self.add(gt, pred, channel_order)

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
