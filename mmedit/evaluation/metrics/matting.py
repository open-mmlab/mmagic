# Copyright (c) OpenMMLab. All rights reserved.
"""Evaluation metrics used in Image Matting."""

from typing import List, Sequence

import cv2
import numpy as np
from mmengine.evaluator import BaseMetric

from mmedit.registry import METRICS
from ..functional import gauss_gradient
from .metrics_utils import average


def _assert_ndim(input, name, ndim, shape_hint):
    if input.ndim != ndim:
        raise ValueError(
            f'{name} should be of shape {shape_hint}, but got {input.shape}.')


def _assert_masked(pred_alpha, trimap):
    if (pred_alpha[trimap == 0] != 0).any() or (pred_alpha[trimap == 255] !=
                                                255).any():
        raise ValueError(
            'pred_alpha should be masked by trimap before evaluation')


def _fetch_data_and_check(data_samples):
    """Fetch and check data from one item of data_batch and predictions.

    Args:
        data_batch (dict): One item of data_batch.
        predictions (dict): One item of predictions.

    Returns:
        pred_alpha (Tensor): Pred_alpha data of predictions.
        ori_alpha (Tensor): Ori_alpha data of data_batch.
        ori_trimap (Tensor): Ori_trimap data of data_batch.
    """
    ori_trimap = data_samples['ori_trimap'][:, :, 0]
    ori_alpha = data_samples['ori_alpha'][:, :, 0]
    pred_alpha = data_samples['output']['pred_alpha']['data']  # 2D tensor
    pred_alpha = pred_alpha.cpu().numpy()

    _assert_ndim(ori_trimap, 'trimap', 2, 'HxW')
    _assert_ndim(ori_alpha, 'gt_alpha', 2, 'HxW')
    _assert_ndim(pred_alpha, 'pred_alpha', 2, 'HxW')
    _assert_masked(pred_alpha, ori_trimap)

    # dtype uint8 -> float64
    pred_alpha = pred_alpha / 255.0
    ori_alpha = ori_alpha / 255.0
    # test shows that using float32 vs float64 differs final results at 1e-4
    # speed are comparable, so we choose float64 for accuracy

    return pred_alpha, ori_alpha, ori_trimap


@METRICS.register_module()
class SAD(BaseMetric):
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

    def __init__(
        self,
        norm_const=1000,
        **kwargs,
    ) -> None:
        self.norm_const = norm_const
        super().__init__(**kwargs)

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


@METRICS.register_module()
class MattingMSE(BaseMetric):
    """Mean Squared Error metric for image matting.

    This metric compute per-pixel squared error average across all
    pixels.
    i.e. mean((a-b)^2) / norm_const

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
        - MattingMSE (float): Mean of Squared Error
    """

    default_prefix = ''

    def __init__(
        self,
        norm_const=1000,
        **kwargs,
    ) -> None:
        self.norm_const = norm_const
        super().__init__(**kwargs)

    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]) -> None:
        """Process one batch of data and predictions.

        Args:
            data_batch (Sequence[dict]): A batch of data
                from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """
        for data_sample in data_samples:
            pred_alpha, gt_alpha, trimap = _fetch_data_and_check(data_sample)

            weight_sum = (trimap == 128).sum()
            if weight_sum != 0:
                mse_result = ((pred_alpha - gt_alpha)**2).sum() / weight_sum
            else:
                mse_result = 0

            self.results.append({'mse': mse_result})

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (dict): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """

        mse = average(results, 'mse')

        return {'MattingMSE': mse}


@METRICS.register_module()
class GradientError(BaseMetric):
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

    def __init__(
        self,
        sigma=1.4,
        norm_constant=1000,
        **kwargs,
    ) -> None:
        self.sigma = sigma
        self.norm_constant = norm_constant
        super().__init__(**kwargs)

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


@METRICS.register_module()
class ConnectivityError(BaseMetric):
    """Connectivity error for evaluating alpha matte prediction.

    .. note::

        Current implementation assume image / alpha / trimap array in numpy
        format and with pixel value ranging from 0 to 255.

    .. note::

        pred_alpha should be masked by trimap before passing
        into this metric

    Args:
        step (float): Step of threshold when computing intersection between
            `alpha` and `pred_alpha`. Default to 0.1 .
        norm_const (int): Divide the result to reduce its magnitude.
            Default to 1000.

    Default prefix: ''

    Metrics:
        - ConnectivityError (float): Connectivity Error
    """

    def __init__(
        self,
        step=0.1,
        norm_constant=1000,
        **kwargs,
    ) -> None:
        self.step = step
        self.norm_constant = norm_constant
        super().__init__(**kwargs)

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

            thresh_steps = np.arange(0, 1 + self.step, self.step)
            round_down_map = -np.ones_like(gt_alpha)
            for i in range(1, len(thresh_steps)):
                gt_alpha_thresh = gt_alpha >= thresh_steps[i]
                pred_alpha_thresh = pred_alpha >= thresh_steps[i]
                intersection = gt_alpha_thresh & pred_alpha_thresh
                intersection = intersection.astype(np.uint8)

                # connected components
                _, output, stats, _ = cv2.connectedComponentsWithStats(
                    intersection, connectivity=4)
                # start from 1 in dim 0 to exclude background
                size = stats[1:, -1]

                # largest connected component of the intersection
                omega = np.zeros_like(gt_alpha)
                if len(size) != 0:
                    max_id = np.argmax(size)
                    # plus one to include background
                    omega[output == max_id + 1] = 1

                mask = (round_down_map == -1) & (omega == 0)
                round_down_map[mask] = thresh_steps[i - 1]
            round_down_map[round_down_map == -1] = 1

            gt_alpha_diff = gt_alpha - round_down_map
            pred_alpha_diff = pred_alpha - round_down_map
            # only calculate difference larger than or equal to 0.15
            gt_alpha_phi = 1 - gt_alpha_diff * (gt_alpha_diff >= 0.15)
            pred_alpha_phi = 1 - pred_alpha_diff * (pred_alpha_diff >= 0.15)

            connectivity_error = np.sum(
                np.abs(gt_alpha_phi - pred_alpha_phi) * (trimap == 128))

            # divide by 1000 to reduce the magnitude of the result
            connectivity_error /= self.norm_constant

            self.results.append({'conn_err': connectivity_error})

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (dict): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """

        conn_err = average(results, 'conn_err')

        return {'ConnectivityError': conn_err}
