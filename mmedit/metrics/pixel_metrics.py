# Copyright (c) OpenMMLab. All rights reserved.
"""Evaluation metrics based on pixels"""

from typing import Optional, Sequence

import numpy as np

from mmedit.registry import METRICS
from .base_sample_wise_metric import BaseSampleWiseMetric
from .utils import img_transform, obtain_data


@METRICS.register_module()
class MAE(BaseSampleWiseMetric):
    """Mean Absolute Error metric for image.

    mean(abs(a-b))

    Args:

        gt_key (str): Key of ground-truth. Default: 'gt_img'
        pred_key (str): Key of prediction. Default: 'pred_img'
        mask_key (str, optional): Key of mask, if mask_key is None, calculate
            all regions. Default: None
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Default: None

    Metrics:
        - MAE (float): Mean of Absolute Error
    """

    metric = 'MAE'

    def process(self, data_batch: Sequence[dict],
                predictions: Sequence[dict]) -> None:
        """Process one batch of data and predictions

        Args:
            data_batch (Sequence[Tuple[Any, dict]]): A batch of data
                from the dataloader.
            predictions (Sequence[dict]): A batch of outputs from
                the model.
        """

        for data, prediction in zip(data_batch, predictions):

            gt = obtain_data(data, self.gt_key)
            pred = obtain_data(prediction, self.pred_key)

            gt = gt / 255.
            pred = pred / 255.

            diff = gt - pred
            diff = abs(diff)

            if self.mask_key is not None:
                mask = obtain_data(data, self.mask_key)
                mask[mask != 0] = 1
                diff *= mask
                result = diff.sum() / mask.sum()
            else:
                result = diff.mean()

            self.results.append({self.metric: result})


@METRICS.register_module()
class MSE(BaseSampleWiseMetric):
    """Mean Squared Error metric for image.

    mean((a-b)^2)

    Args:

        gt_key (str): Key of ground-truth. Default: 'gt_img'
        pred_key (str): Key of prediction. Default: 'pred_img'
        mask_key (str, optional): Key of mask, if mask_key is None, calculate
            all regions. Default: None
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Default: None

    Metrics:
        - MSE (float): Mean of Squared Error
    """

    metric = 'MSE'

    def process(self, data_batch: Sequence[dict],
                predictions: Sequence[dict]) -> None:
        """Process one batch of data and predictions

        Args:
            data_batch (Sequence[Tuple[Any, dict]]): A batch of data
                from the dataloader.
            predictions (Sequence[dict]): A batch of outputs from
                the model.
        """

        for data, prediction in zip(data_batch, predictions):

            gt = obtain_data(data, self.gt_key)
            pred = obtain_data(prediction, self.pred_key)

            gt = gt / 255.
            pred = pred / 255.

            diff = gt - pred
            diff *= diff

            if self.mask_key is not None:
                mask = obtain_data(data, self.mask_key)
                mask[mask != 0] = 1
                diff *= mask
                result = diff.sum() / mask.sum()
            else:
                result = diff.mean()

            self.results.append({self.metric: result})


@METRICS.register_module()
class PSNR(BaseSampleWiseMetric):
    """Peak Signal-to-Noise Ratio.

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

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
            pixels are not involved in the PSNR calculation. Default: 0.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'CHW'.
        convert_to (str): Whether to convert the images to other color models.
            If None, the images are not altered. When computing for 'Y',
            the images are assumed to be in BGR order. Options are 'Y' and
            None. Default: None.

    Metrics:
        - PSNR (float): Peak Signal-to-Noise Ratio
    """

    metric = 'PSNR'

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

    def process(self, data_batch: Sequence[dict],
                predictions: Sequence[dict]) -> None:
        """Process one batch of data and predictions

        Args:
            data_batch (Sequence[Tuple[Any, dict]]): A batch of data
                from the dataloader.
            predictions (Sequence[dict]): A batch of outputs from
                the model.
        """

        for data, prediction in zip(data_batch, predictions):

            gt = obtain_data(data, self.gt_key)
            pred = obtain_data(prediction, self.pred_key)

            result = psnr(
                img1=gt,
                img2=pred,
                crop_border=self.crop_border,
                input_order=self.input_order,
                convert_to=self.convert_to)

            self.results.append({self.metric: result})


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
            pixels are not involved in the PSNR calculation. Default: 0.
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

    def process(self, data_batch: Sequence[dict],
                predictions: Sequence[dict]) -> None:
        """Process one batch of data and predictions

        Args:
            data_batch (Sequence[Tuple[Any, dict]]): A batch of data
                from the dataloader.
            predictions (Sequence[dict]): A batch of outputs from
                the model.
        """

        for data, prediction in zip(data_batch, predictions):

            gt = obtain_data(data, self.gt_key)
            pred = obtain_data(prediction, self.pred_key)

            result = snr(
                gt=gt,
                pred=pred,
                crop_border=self.crop_border,
                input_order=self.input_order,
                convert_to=self.convert_to)

            self.results.append({self.metric: result})


def psnr(img1, img2, crop_border=0, input_order='HWC', convert_to=None):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edges of an image. These
            pixels are not involved in the PSNR calculation. Default: 0.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        convert_to (str): Whether to convert the images to other color models.
            If None, the images are not altered. When computing for 'Y',
            the images are assumed to be in BGR order. Options are 'Y' and
            None. Default: None.

    Returns:
        float: psnr result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are different: {img1.shape}, {img2.shape}.')

    img1 = img_transform(
        img1,
        crop_border=crop_border,
        input_order=input_order,
        convert_to=convert_to)
    img2 = img_transform(
        img2,
        crop_border=crop_border,
        input_order=input_order,
        convert_to=convert_to)

    mse_value = ((img1 - img2)**2).mean()
    if mse_value == 0:
        result = float('inf')
    else:
        result = 20. * np.log10(255. / np.sqrt(mse_value))

    return result


def snr(gt, pred, crop_border=0, input_order='HWC', convert_to=None):
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

    Returns:
        float: psnr result.
    """

    assert gt.shape == pred.shape, (
        f'Image shapes are different: {gt.shape}, {pred.shape}.')

    gt = img_transform(
        gt,
        crop_border=crop_border,
        input_order=input_order,
        convert_to=convert_to)
    pred = img_transform(
        pred,
        crop_border=crop_border,
        input_order=input_order,
        convert_to=convert_to)

    signal = ((gt)**2).mean()
    noise = ((gt - pred)**2).mean()

    result = 10. * np.log10(signal / noise)

    return result
