# Copyright (c) OpenMMLab. All rights reserved.
import math
import os
from typing import Optional

import cv2
import mmcv
import numpy as np
from scipy.ndimage import convolve
from scipy.special import gamma

from mmedit.datasets.transforms import MATLABLikeResize
from mmedit.registry import METRICS
from mmedit.utils import reorder_image, to_numpy
from .base_sample_wise_metric import BaseSampleWiseMetric


@METRICS.register_module()
class NIQE(BaseSampleWiseMetric):
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
            Default: 'HWC'.
        convert_to (str): Whether to convert the images to other color models.
            If None, the images are not altered. When computing for 'Y',
            the images are assumed to be in BGR order. Options are 'Y' and
            None. Default: 'gray'.

    Metrics:
        - NIQE (float): Natural Image Quality Evaluator
    """

    metric = 'NIQE'

    def __init__(self,
                 key: str = 'pred_img',
                 is_predicted: bool = True,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 crop_border=0,
                 input_order='HWC',
                 convert_to='gray') -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        convert_to = convert_to.lower()
        assert convert_to in [
            'y', 'gray'
        ], ('Only support gray image, '
            "``convert_to`` should be selected from ['y', 'gray']")

        self.key = key
        self.is_predicted = is_predicted
        self.crop_border = crop_border
        self.input_order = input_order
        self.convert_to = convert_to

    def process_image(self, gt, pred, mask) -> None:
        """Process an image.

        Args:
            gt (np.ndarray): GT image.
            pred (np.ndarray): Pred image.
            mask (np.ndarray): Mask of evaluation.
        Returns:
            result (np.ndarray): NIQE result.
        """

        result = niqe(
            img=pred,
            crop_border=self.crop_border,
            input_order=self.input_order,
            convert_to=self.convert_to)
        return result


def estimate_aggd_param(block):
    """Estimate AGGD (Asymmetric Generalized Gaussian Distribution) parameters.

    Args:
        block (np.ndarray): 2D Image block.

    Returns:
        tuple: alpha (float), beta_l (float) and beta_r (float) for the AGGD
            distribution (Estimating the parames in Equation 7 in the paper).
    """
    block = block.flatten()
    gam = np.arange(0.2, 10.001, 0.001)  # len = 9801
    gam_reciprocal = np.reciprocal(gam)
    r_gam = np.square(gamma(gam_reciprocal * 2)) / (
        gamma(gam_reciprocal) * gamma(gam_reciprocal * 3))

    left_std = np.sqrt(np.mean(block[block < 0]**2))
    right_std = np.sqrt(np.mean(block[block > 0]**2))
    gammahat = left_std / right_std
    rhat = (np.mean(np.abs(block)))**2 / np.mean(block**2)
    rhatnorm = (rhat * (gammahat**3 + 1) *
                (gammahat + 1)) / ((gammahat**2 + 1)**2)
    array_position = np.argmin((r_gam - rhatnorm)**2)

    alpha = gam[array_position]
    beta_l = left_std * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))
    beta_r = right_std * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))
    return (alpha, beta_l, beta_r)


def compute_feature(block):
    """Compute features.

    Args:
        block (np.ndarray): 2D Image block.

    Returns:
        feat (List): Features with length of 18.
    """
    feat = []
    alpha, beta_l, beta_r = estimate_aggd_param(block)
    feat.extend([alpha, (beta_l + beta_r) / 2])

    # distortions disturb the fairly regular structure of natural images.
    # This deviation can be captured by analyzing the sample distribution of
    # the products of pairs of adjacent coefficients computed along
    # horizontal, vertical and diagonal orientations.
    shifts = [[0, 1], [1, 0], [1, 1], [1, -1]]
    for shift in shifts:
        shifted_block = np.roll(block, shift, axis=(0, 1))
        alpha, beta_l, beta_r = estimate_aggd_param(block * shifted_block)
        mean = (beta_r - beta_l) * (gamma(2 / alpha) / gamma(1 / alpha))
        feat.extend([alpha, mean, beta_l, beta_r])
    return feat


def niqe_core(img,
              mu_pris_param,
              cov_pris_param,
              gaussian_window,
              block_size_h=96,
              block_size_w=96):
    """Calculate NIQE (Natural Image Quality Evaluator) metric.

    Ref: Making a "Completely Blind" Image Quality Analyzer.
    This implementation could produce almost the same results as the official
    MATLAB codes: http://live.ece.utexas.edu/research/quality/niqe_release.zip

    Note that we do not include block overlap height and width, since they are
    always 0 in the official implementation.

    For good performance, it is advisable by the official implementation to
    divide the distorted image in to the same size patched as used for the
    construction of multivariate Gaussian model.

    Args:
        img (np.ndarray): Input image whose quality needs to be computed. The
            image must be a gray or Y (of YCbCr) image with shape (h, w).
            Range [0, 255] with float type.
        mu_pris_param (np.ndarray): Mean of a pre-defined multivariate Gaussian
            model calculated on the pristine dataset.
        cov_pris_param (np.ndarray): Covariance of a pre-defined multivariate
            Gaussian model calculated on the pristine dataset.
        gaussian_window (ndarray): A 7x7 Gaussian window used for smoothing the
            image.
        block_size_h (int): Height of the blocks in to which image is divided.
            Default: 96 (the official recommended value). Default: 96.
        block_size_w (int): Width of the blocks in to which image is divided.
            Default: 96 (the official recommended value). Default: 96.

    Returns:
        np.ndarray: NIQE quality.
    """
    # crop image
    h, w = img.shape
    num_block_h = math.floor(h / block_size_h)
    num_block_w = math.floor(w / block_size_w)
    img = img[0:num_block_h * block_size_h, 0:num_block_w * block_size_w]

    distparam = []  # dist param is actually the multiscale features
    for scale in (1, 2):  # perform on two scales (1, 2)
        mu = convolve(img, gaussian_window, mode='nearest')

        sigma = np.sqrt(
            np.abs(
                convolve(np.square(img), gaussian_window, mode='nearest') -
                np.square(mu)))
        # normalize, as in Eq. 1 in the paper
        img_nomalized = (img - mu) / (sigma + 1)

        feat = []
        for idx_w in range(num_block_w):
            for idx_h in range(num_block_h):
                # process each block
                block = img_nomalized[idx_h * block_size_h //
                                      scale:(idx_h + 1) * block_size_h //
                                      scale, idx_w * block_size_w //
                                      scale:(idx_w + 1) * block_size_w //
                                      scale]
                feat.append(compute_feature(block))

        distparam.append(np.array(feat))

        # matlab-like bicubic downsample with anti-aliasing
        if scale == 1:
            resize = MATLABLikeResize(keys=None, scale=0.5)
            img = resize._resize(img[:, :, np.newaxis] / 255.)[:, :, 0] * 255.

    distparam = np.concatenate(distparam, axis=1)

    # fit a MVG (multivariate Gaussian) model to distorted patch features
    mu_distparam = np.nanmean(distparam, axis=0)
    distparam_no_nan = distparam[~np.isnan(distparam).any(axis=1)]
    cov_distparam = np.cov(distparam_no_nan, rowvar=False)

    # compute niqe quality, Eq. 10 in the paper
    invcov_param = np.linalg.pinv((cov_pris_param + cov_distparam) / 2)
    quality = np.matmul(
        np.matmul((mu_pris_param - mu_distparam), invcov_param),
        np.transpose((mu_pris_param - mu_distparam)))

    return np.squeeze(np.sqrt(quality))


def niqe(img, crop_border, input_order='HWC', convert_to='y'):
    """Calculate NIQE (Natural Image Quality Evaluator) metric.

    Ref: Making a "Completely Blind" Image Quality Analyzer.
    This implementation could produce almost the same results as the official
    MATLAB codes: http://live.ece.utexas.edu/research/quality/niqe_release.zip

    We use the official params estimated from the pristine dataset.
    We use the recommended block size (96, 96) without overlaps.

    Args:
        img (np.ndarray): Input image whose quality needs to be computed.
            The input image must be in range [0, 255] with float/int type.
            The input_order of image can be 'HW' or 'HWC' or 'CHW'. (BGR order)
            If the input order is 'HWC' or 'CHW', it will be converted to gray
            or Y (of YCbCr) image according to the ``convert_to`` argument.
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the metric calculation.
        input_order (str): Whether the input order is 'HW', 'HWC' or 'CHW'.
            Default: 'HWC'.
        convert_to (str): Whether converted to 'y' (of MATLAB YCbCr) or 'gray'.
            Default: 'y'.

    Returns:
        niqe_result (float): NIQE result.
    """

    # we use the official params estimated from the pristine dataset.
    niqe_pris_params = np.load(
        os.path.join(os.path.dirname(__file__), 'niqe_pris_params.npz'))
    mu_pris_param = niqe_pris_params['mu_pris_param']
    cov_pris_param = niqe_pris_params['cov_pris_param']
    gaussian_window = niqe_pris_params['gaussian_window']

    img = to_numpy(img, np.float32)

    if input_order != 'HW':
        img = reorder_image(img, input_order=input_order)
        if convert_to == 'y':
            img = mmcv.bgr2ycbcr(img / 255., y_only=True) * 255.
        elif convert_to == 'gray':
            img = mmcv.bgr2gray(img / 255., cv2.COLOR_BGR2GRAY) * 255.
        img = np.squeeze(img)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border]

    # round to follow official implementation
    img = img.round()

    niqe_result = niqe_core(img, mu_pris_param, cov_pris_param,
                            gaussian_window)

    return niqe_result
