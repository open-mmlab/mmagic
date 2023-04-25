# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import List, Optional, Sequence

import numpy as np
import torch
from mmengine.dist import all_gather, get_world_size, is_main_process
from scipy import signal

from mmagic.registry import METRICS
from .base_gen_metric import GenerativeMetric


def _f_special_gauss(size, sigma):
    r"""Return a circular symmetric gaussian kernel.

    Ref: https://github.com/tkarras/progressive_growing_of_gans/blob/master/metrics/ms_ssim.py  # noqa

    Args:
        size (int): Size of Gaussian kernel.
        sigma (float): Standard deviation for Gaussian blur kernel.

    Returns:
        ndarray: Gaussian kernel.
    """
    radius = size // 2
    offset = 0.0
    start, stop = -radius, radius + 1
    if size % 2 == 0:
        offset = 0.5
        stop -= 1
    x, y = np.mgrid[offset + start:stop, offset + start:stop]
    assert len(x) == size
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    return g / g.sum()


def _hox_downsample(img):
    r"""Downsample images with factor equal to 0.5.

    Ref: https://github.com/tkarras/progressive_growing_of_gans/blob/master/metrics/ms_ssim.py  # noqa

    Args:
        img (ndarray): Images with order "NHWC".

    Returns:
        ndarray: Downsampled images with order "NHWC".
    """
    return (img[:, 0::2, 0::2, :] + img[:, 1::2, 0::2, :] +
            img[:, 0::2, 1::2, :] + img[:, 1::2, 1::2, :]) * 0.25


def _ssim_for_multi_scale(img1,
                          img2,
                          max_val=255,
                          filter_size=11,
                          filter_sigma=1.5,
                          k1=0.01,
                          k2=0.03):
    """Calculate SSIM (structural similarity) and contrast sensitivity.

    Ref:
    Image quality assessment: From error visibility to structural similarity.

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    This function attempts to match the functionality of ssim_index_new.m by
    Zhou Wang: http://www.cns.nyu.edu/~lcv/ssim/msssim.zip

    Args:
        img1 (ndarray): Images with range [0, 255] and order "NHWC".
        img2 (ndarray): Images with range [0, 255] and order "NHWC".
        max_val (int): the dynamic range of the images (i.e., the difference
            between the maximum the and minimum allowed values).
            Default to 255.
        filter_size (int): Size of blur kernel to use (will be reduced for
            small images). Default to 11.
        filter_sigma (float): Standard deviation for Gaussian blur kernel (will
            be reduced for small images). Default to 1.5.
        k1 (float): Constant used to maintain stability in the SSIM calculation
            (0.01 in the original paper). Default to 0.01.
        k2 (float): Constant used to maintain stability in the SSIM calculation
            (0.03 in the original paper). Default to 0.03.

    Returns:
        tuple: Pair containing the mean SSIM and contrast sensitivity between
        `img1` and `img2`.
    """
    if img1.shape != img2.shape:
        raise RuntimeError(
            'Input images must have the same shape (%s vs. %s).' %
            (img1.shape, img2.shape))
    if img1.ndim != 4:
        raise RuntimeError('Input images must have four dimensions, not %d' %
                           img1.ndim)

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    _, height, width, _ = img1.shape

    # Filter size can't be larger than height or width of images.
    size = min(filter_size, height, width)

    # Scale down sigma if a smaller filter size is used.
    sigma = size * filter_sigma / filter_size if filter_size else 0

    if filter_size:
        window = np.reshape(_f_special_gauss(size, sigma), (1, size, size, 1))
        mu1 = signal.fftconvolve(img1, window, mode='valid')
        mu2 = signal.fftconvolve(img2, window, mode='valid')
        sigma11 = signal.fftconvolve(img1 * img1, window, mode='valid')
        sigma22 = signal.fftconvolve(img2 * img2, window, mode='valid')
        sigma12 = signal.fftconvolve(img1 * img2, window, mode='valid')
    else:
        # Empty blur kernel so no need to convolve.
        mu1, mu2 = img1, img2
        sigma11 = img1 * img1
        sigma22 = img2 * img2
        sigma12 = img1 * img2

    mu11 = mu1 * mu1
    mu22 = mu2 * mu2
    mu12 = mu1 * mu2
    sigma11 -= mu11
    sigma22 -= mu22
    sigma12 -= mu12

    # Calculate intermediate values used by both ssim and cs_map.
    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    v1 = 2.0 * sigma12 + c2
    v2 = sigma11 + sigma22 + c2
    ssim = np.mean((((2.0 * mu12 + c1) * v1) / ((mu11 + mu22 + c1) * v2)),
                   axis=(1, 2, 3))  # Return for each image individually.
    cs = np.mean(v1 / v2, axis=(1, 2, 3))
    return ssim, cs


def ms_ssim(img1,
            img2,
            max_val=255,
            filter_size=11,
            filter_sigma=1.5,
            k1=0.01,
            k2=0.03,
            weights=None,
            reduce_mean=True) -> np.ndarray:
    """Calculate MS-SSIM (multi-scale structural similarity).

    Ref:
    This function implements Multi-Scale Structural Similarity (MS-SSIM) Image
    Quality Assessment according to Zhou Wang's paper, "Multi-scale structural
    similarity for image quality assessment" (2003).
    Link: https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf

    Author's MATLAB implementation:
    http://www.cns.nyu.edu/~lcv/ssim/msssim.zip

    PGGAN's implementation:
    https://github.com/tkarras/progressive_growing_of_gans/blob/master/metrics/ms_ssim.py

    Args:
        img1 (ndarray): Images with range [0, 255] and order "NHWC".
        img2 (ndarray): Images with range [0, 255] and order "NHWC".
        max_val (int): the dynamic range of the images (i.e., the difference
            between the maximum the and minimum allowed values).
            Default to 255.
        filter_size (int): Size of blur kernel to use (will be reduced for
            small images). Default to 11.
        filter_sigma (float): Standard deviation for Gaussian blur kernel (will
            be reduced for small images). Default to 1.5.
        k1 (float): Constant used to maintain stability in the SSIM calculation
            (0.01 in the original paper). Default to 0.01.
        k2 (float): Constant used to maintain stability in the SSIM calculation
            (0.03 in the original paper). Default to 0.03.
        weights (list): List of weights for each level; if none, use five
            levels and the weights from the original paper. Default to None.

    Returns:
        np.ndarray: MS-SSIM score between `img1` and `img2`.
    """
    if img1.shape != img2.shape:
        raise RuntimeError(
            'Input images must have the same shape (%s vs. %s).' %
            (img1.shape, img2.shape))
    if img1.ndim != 4:
        raise RuntimeError('Input images must have four dimensions, not %d' %
                           img1.ndim)

    # Note: default weights don't sum to 1.0 but do match the paper / matlab
    # code.
    weights = np.array(
        weights if weights else [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    levels = weights.size
    im1, im2 = [x.astype(np.float32) for x in [img1, img2]]
    mssim = []
    mcs = []
    for _ in range(levels):
        ssim, cs = _ssim_for_multi_scale(
            im1,
            im2,
            max_val=max_val,
            filter_size=filter_size,
            filter_sigma=filter_sigma,
            k1=k1,
            k2=k2)
        mssim.append(ssim)
        mcs.append(cs)
        im1, im2 = [_hox_downsample(x) for x in [im1, im2]]

    # Clip to zero. Otherwise we get NaNs.
    mssim = np.clip(np.asarray(mssim), 0.0, np.inf)
    mcs = np.clip(np.asarray(mcs), 0.0, np.inf)

    results = np.prod(mcs[:-1, :]**weights[:-1, np.newaxis], axis=0) * \
        (mssim[-1, :]**weights[-1])
    if reduce_mean:
        # Average over images only at the end.
        results = np.mean(results)
    return results


@METRICS.register_module('MS_SSIM')
@METRICS.register_module()
class MultiScaleStructureSimilarity(GenerativeMetric):
    """MS-SSIM (Multi-Scale Structure Similarity) metric.

    Ref: https://github.com/tkarras/progressive_growing_of_gans/blob/master/metrics/ms_ssim.py # noqa

    Args:
        fake_nums (int): Numbers of the generated image need for the metric.
        fake_key (Optional[str]): Key for get fake images of the output dict.
            Defaults to None.
        real_key (Optional[str]): Key for get real images from the input dict.
            Defaults to 'img'.
        need_cond_input (bool): If true, the sampler will return the
            conditional input randomly sampled from the original dataset.
            This require the dataset implement `get_data_info` and field
            `gt_label` must be contained in the return value of
            `get_data_info`. Noted that, for unconditional models, set
            `need_cond_input` as True may influence the result of evaluation
            results since the conditional inputs are sampled from the dataset
            distribution; otherwise will be sampled from the uniform
            distribution. Defaults to False.

        sample_model (str): Sampling mode for the generative model. Support
            'orig' and 'ema'. Defaults to 'ema'.
        collect_device (str, optional): Device name used for collecting results
            from different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """
    name = 'MS-SSIM'

    def __init__(self,
                 fake_nums: int,
                 fake_key: Optional[str] = None,
                 need_cond_input: bool = False,
                 sample_model: str = 'ema',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(fake_nums, 0, fake_key, None, need_cond_input,
                         sample_model, collect_device, prefix)

        assert fake_nums % 2 == 0
        self.num_pairs = fake_nums // 2

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Feed data to the metric.

        Args:
            data_batch (dict): Real images from dataloader. Do not be used
                in this metric.
            data_samples (Sequence[dict]): Generated images.
        """
        if len(self.fake_results) >= (self.fake_nums_per_device // 2):
            return

        fake_imgs = []
        for pred in data_samples:
            fake_img_ = pred
            # get ema/orig results
            if self.sample_model in fake_img_:
                fake_img_ = fake_img_[self.sample_model]
            # get specific fake_keys
            if (self.fake_key is not None and self.fake_key in fake_img_):
                fake_img_ = fake_img_[self.fake_key]
            else:
                # get img tensor
                fake_img_ = fake_img_['fake_img']
            fake_imgs.append(fake_img_)
        minibatch = torch.stack(fake_imgs, dim=0)

        assert minibatch.shape[0] % 2 == 0, 'batch size must be divided by 2.'

        half1 = minibatch[0::2].cpu().data.numpy().transpose((0, 2, 3, 1))
        half2 = minibatch[1::2].cpu().data.numpy().transpose((0, 2, 3, 1))

        scores = ms_ssim(half1, half2, reduce_mean=False)
        self.fake_results += [torch.Tensor([s]) for s in scores.tolist()]

    def _collect_target_results(self, target: str) -> Optional[list]:
        """Collected results for MS-SSIM metric. Size of `self.fake_results` in
        MS-SSIM does not relay on `self.fake_nums` but `self.num_pairs`.

        Args:
            target (str): Target results to collect.

        Returns:
            Optional[list]: The collected results.
        """
        assert target in 'fake', 'Only support to collect \'fake\' results.'
        results = getattr(self, f'{target}_results')
        size = self.num_pairs
        size = len(results) * get_world_size() if size == -1 else size

        if len(results) == 0:
            warnings.warn(
                f'{self.__class__.__name__} got empty `self.{target}_results`.'
                ' Please ensure that the processed results are properly added '
                f'into `self.{target}_results` in `process` method.')

        # apply all_gather for tensor results
        results = torch.cat(results, dim=0)
        results = torch.cat(all_gather(results), dim=0)[:size]
        results = torch.split(results, 1)

        # on non-main process, results should be `None`
        if is_main_process() and len(results) != size:
            raise ValueError(f'Length of results is \'{len(results)}\', not '
                             f'equals to target size \'{size}\'.')
        return results

    def compute_metrics(self, results_fake: List):
        """Computed the result of MS-SSIM.

        Returns:
            dict: Calculated MS-SSIM result.
        """
        results = torch.cat(results_fake, dim=0)
        avg = results.sum() / self.num_pairs
        return {'avg': round(avg.item(), 4)}
