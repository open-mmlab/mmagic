# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import numpy as np

from mmagic.registry import METRICS
from .base_sample_wise_metric import BaseSampleWiseMetric
from .metrics_utils import img_transform


@METRICS.register_module()
class SNR(BaseSampleWiseMetric):
    """Signal-to-Noise Ratio.

    Ref: https://en.wikipedia.org/wiki/Signal-to-noise_ratio

    Args:

        gt_key (str): Key of ground-truth. Default: 'gt_img'
        pred_key (str): Key of prediction. Default: 'pred_img'
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Default: None
        crop_border (int): Cropped pixels in each edges of an image. These
            pixels are not involved in the SNR calculation. Default: 0.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'CHW'.
        convert_to (str): Whether to convert the images to other color models.
            If None, the images are not altered. When computing for 'Y',
            the images are assumed to be in BGR order. Options are 'Y' and
            None. Default: None.

    Metrics:
        - SNR (float): Signal-to-Noise Ratio
    """

    metric = 'SNR'

    def __init__(self,
                 gt_key: str = 'gt_img',
                 pred_key: str = 'pred_img',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 crop_border=0,
                 input_order='CHW',
                 convert_to=None) -> None:
        super().__init__(
            gt_key=gt_key,
            pred_key=pred_key,
            mask_key=None,
            collect_device=collect_device,
            prefix=prefix)

        self.crop_border = crop_border
        self.input_order = input_order
        self.convert_to = convert_to

    def process_image(self, gt, pred, mask):
        """Process an image.

        Args:
            gt (Torch | np.ndarray): GT image.
            pred (Torch | np.ndarray): Pred image.
            mask (Torch | np.ndarray): Mask of evaluation.
        Returns:
            np.ndarray: SNR result.
        """

        return snr(
            gt=gt,
            pred=pred,
            crop_border=self.crop_border,
            input_order=self.input_order,
            convert_to=self.convert_to,
            channel_order=self.channel_order)


def snr(gt,
        pred,
        crop_border=0,
        input_order='HWC',
        convert_to=None,
        channel_order='rgb'):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        gt (ndarray): Images with range [0, 255].
        pred (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edges of an image. These
            pixels are not involved in the PSNR calculation. Default: 0.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        convert_to (str): Whether to convert the images to other color models.
            If None, the images are not altered. When computing for 'Y',
            the images are assumed to be in BGR order. Options are 'Y' and
            None. Default: None.
        channel_order (str): The channel order of image. Default: 'rgb'.

    Returns:
        float: SNR result.
    """

    assert gt.shape == pred.shape, (
        f'Image shapes are different: {gt.shape}, {pred.shape}.')

    gt = img_transform(
        gt,
        crop_border=crop_border,
        input_order=input_order,
        convert_to=convert_to,
        channel_order=channel_order)
    pred = img_transform(
        pred,
        crop_border=crop_border,
        input_order=input_order,
        convert_to=convert_to,
        channel_order=channel_order)

    signal = ((gt)**2).mean()
    noise = ((gt - pred)**2).mean()

    result = 10. * np.log10(signal / noise)

    return result
