# Copyright (c) OpenMMLab. All rights reserved.
"""Augmentation on trimaps."""

import cv2
import numpy as np
from mmcv.transforms import BaseTransform
from mmengine.utils import is_tuple_of

from mmedit.registry import TRANSFORMS


@TRANSFORMS.register_module()
class FormatTrimap(BaseTransform):
    """Convert trimap (tensor) to one-hot representation.

    It transforms the trimap label from (0, 128, 255) to (0, 1, 2). If
    ``to_onehot`` is set to True, the trimap will convert to one-hot tensor of
    shape (3, H, W). Required key is "trimap", added or modified key are
    "trimap" and "format_trimap_to_onehot".

    Args:
        to_onehot (bool): whether convert trimap to one-hot tensor. Default:
            ``False``.
    """

    def __init__(self, to_onehot=False):
        self.to_onehot = to_onehot

    def transform(self, results):
        """Transform function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        trimap = results['trimap'].squeeze()
        assert trimap.ndim == 2

        if self.to_onehot:
            trimap_one_hot = np.zeros((*trimap.shape, 3), dtype=np.uint8)
            trimap_one_hot[..., 0][trimap == 0] = 1
            trimap_one_hot[..., 1][trimap == 128] = 1
            trimap_one_hot[..., 2][trimap == 255] = 1
            results['trimap'] = trimap_one_hot
        else:
            trimap[trimap == 128] = 1
            trimap[trimap == 255] = 2
            results['trimap'] = trimap

        results['format_trimap_to_onehot'] = self.to_onehot
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(to_onehot={self.to_onehot})'


@TRANSFORMS.register_module()
class GenerateTrimap(BaseTransform):
    """Using random erode/dilate to generate trimap from alpha matte.

    Required key is "alpha", added key is "trimap".

    Args:
        kernel_size (int | tuple[int]): The range of random kernel_size of
            erode/dilate; int indicates a fixed kernel_size. If `random` is set
            to False and kernel_size is a tuple of length 2, then it will be
            interpreted as (erode kernel_size, dilate kernel_size). It should
            be noted that the kernel of the erosion and dilation has the same
            height and width.
        iterations (int | tuple[int], optional): The range of random iterations
            of erode/dilate; int indicates a fixed iterations. If `random` is
            set to False and iterations is a tuple of length 2, then it will be
            interpreted as (erode iterations, dilate iterations). Default to 1.
        random (bool, optional): Whether use random kernel_size and iterations
            when generating trimap. See `kernel_size` and `iterations` for more
            information. Default to True.
    """

    def __init__(self, kernel_size, iterations=1, random=True):
        if isinstance(kernel_size, int):
            kernel_size = kernel_size, kernel_size + 1
        elif not is_tuple_of(kernel_size, int) or len(kernel_size) != 2:
            raise ValueError('kernel_size must be an int or a tuple of 2 int, '
                             f'but got {kernel_size}')

        if isinstance(iterations, int):
            iterations = iterations, iterations + 1
        elif not is_tuple_of(iterations, int) or len(iterations) != 2:
            raise ValueError('iterations must be an int or a tuple of 2 int, '
                             f'but got {iterations}')

        self.random = random
        if self.random:
            min_kernel, max_kernel = kernel_size
            self.iterations = iterations
            self.kernels = [
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
                for size in range(min_kernel, max_kernel)
            ]
        else:
            erode_ksize, dilate_ksize = kernel_size
            self.iterations = iterations
            self.kernels = [
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (erode_ksize, erode_ksize)),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (dilate_ksize, dilate_ksize))
            ]

    def transform(self, results: dict) -> dict:
        """Transform function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        alpha = results['alpha']

        if self.random:
            kernel_num = len(self.kernels)
            erode_kernel_idx = np.random.randint(kernel_num)
            dilate_kernel_idx = np.random.randint(kernel_num)
            min_iter, max_iter = self.iterations
            erode_iter = np.random.randint(min_iter, max_iter)
            dilate_iter = np.random.randint(min_iter, max_iter)
        else:
            erode_kernel_idx, dilate_kernel_idx = 0, 1
            erode_iter, dilate_iter = self.iterations

        eroded = cv2.erode(
            alpha, self.kernels[erode_kernel_idx], iterations=erode_iter)
        dilated = cv2.dilate(
            alpha, self.kernels[dilate_kernel_idx], iterations=dilate_iter)

        trimap = np.zeros_like(alpha)
        trimap.fill(128)
        trimap[eroded >= 255] = 255
        trimap[dilated <= 0] = 0
        results['trimap'] = trimap
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(kernels={self.kernels}, iterations={self.iterations}, '
                     f'random={self.random})')
        return repr_str


@TRANSFORMS.register_module()
class GenerateTrimapWithDistTransform(BaseTransform):
    """Generate trimap with distance transform function.

    Args:
        dist_thr (int, optional): Distance threshold. Area with alpha value
            between (0, 255) will be considered as initial unknown area. Then
            area with distance to unknown area smaller than the distance
            threshold will also be consider as unknown area. Defaults to 20.
        random (bool, optional): If True, use random distance threshold from
            [1, dist_thr). If False, use `dist_thr` as the distance threshold
            directly. Defaults to True.
    """

    def __init__(self, dist_thr=20, random=True):
        if not (isinstance(dist_thr, int) and dist_thr >= 1):
            raise ValueError('dist_thr must be an int that is greater than 1, '
                             f'but got {dist_thr}')
        self.dist_thr = dist_thr
        self.random = random

    def transform(self, results: dict) -> dict:
        """Transform function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        alpha = results['alpha']

        # image dilation implemented by Euclidean distance transform
        known = (alpha == 0) | (alpha == 255)
        dist_to_unknown = cv2.distanceTransform(
            known.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        dist_thr = np.random.randint(
            1, self.dist_thr) if self.random else self.dist_thr
        unknown = dist_to_unknown <= dist_thr

        trimap = (alpha == 255).astype(np.uint8) * 255
        trimap[unknown] = 128
        results['trimap'] = trimap
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(dist_thr={self.dist_thr}, random={self.random})'
        return repr_str


@TRANSFORMS.register_module()
class TransformTrimap(BaseTransform):
    """Transform trimap into two-channel and six-channel.

    This class will generate a two-channel trimap composed of definite
    foreground and background masks and encode it into a six-channel trimap
    using Gaussian blurs of the generated two-channel trimap at three
    different scales. The transformed trimap has 6 channels.

    Required key is "trimap", added key is "transformed_trimap" and
    "two_channel_trimap".

    Adopted from the following repository:
    https://github.com/MarcoForte/FBA_Matting/blob/master/networks/transforms.py.
    """

    def transform(self, results: dict) -> dict:
        """Transform function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        trimap = results['trimap']
        assert len(trimap.shape) == 2
        h, w = trimap.shape[:2]
        # generate two-channel trimap
        trimap2 = np.zeros((h, w, 2), dtype=np.uint8)
        trimap2[trimap == 0, 0] = 255
        trimap2[trimap == 255, 1] = 255
        trimap_trans = np.zeros((h, w, 6), dtype=np.float32)
        factor = np.array([[[0.02, 0.08, 0.16]]], dtype=np.float32)
        for k in range(2):
            if np.any(trimap2[:, :, k]):
                dt_mask = -cv2.distanceTransform(255 - trimap2[:, :, k],
                                                 cv2.DIST_L2, 0)**2
                dt_mask = dt_mask[..., None]
                L = 320
                trimap_trans[..., 3 * k:3 * k +
                             3] = np.exp(dt_mask / (2 * ((factor * L)**2)))

        results['transformed_trimap'] = trimap_trans
        results['two_channel_trimap'] = trimap2
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
