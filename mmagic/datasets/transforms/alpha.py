# Copyright (c) OpenMMLab. All rights reserved.
"""Augmentation on alpha matte."""
# Not used in current algorithms

import random

import cv2
import numpy as np
from mmcv.transforms import BaseTransform
from mmengine.utils import is_list_of, is_tuple_of

from mmagic.registry import TRANSFORMS
from mmagic.utils import random_choose_unknown


@TRANSFORMS.register_module()
class GenerateSeg(BaseTransform):
    """Generate segmentation mask from alpha matte.

    Args:
        kernel_size (int, optional): Kernel size for both erosion and
            dilation. The kernel will have the same height and width.
            Defaults to 5.
        erode_iter_range (tuple, optional): Iteration of erosion.
            Defaults to (10, 20).
        dilate_iter_range (tuple, optional): Iteration of dilation.
            Defaults to (15, 30).
        num_holes_range (tuple, optional): Range of number of holes to
            randomly select from. Defaults to (0, 3).
        hole_sizes (list, optional): List of (h, w) to be selected as the
            size of the rectangle hole.
            Defaults to [(15, 15), (25, 25), (35, 35), (45, 45)].
        blur_ksizes (list, optional): List of (h, w) to be selected as the
            kernel_size of the gaussian blur.
            Defaults to [(21, 21), (31, 31), (41, 41)].
    """

    def __init__(self,
                 kernel_size=5,
                 erode_iter_range=(10, 20),
                 dilate_iter_range=(15, 30),
                 num_holes_range=(0, 3),
                 hole_sizes=[(15, 15), (25, 25), (35, 35), (45, 45)],
                 blur_ksizes=[(21, 21), (31, 31), (41, 41)]):
        self.kernel_size = kernel_size
        self.erode_iter_range = erode_iter_range
        self.dilate_iter_range = dilate_iter_range
        self.num_holes_range = num_holes_range
        self.hole_sizes = hole_sizes
        self.blur_ksizes = blur_ksizes

    @staticmethod
    def _crop_hole(img, start_point, hole_size):
        """Create a all-zero rectangle hole in the image.

        Args:
            img (np.ndarray): Source image.
            start_point (tuple[int]): The top-left point of the rectangle.
            hole_size (tuple[int]): The height and width of the rectangle hole.

        Return:
            np.ndarray: The cropped image.
        """
        top, left = start_point
        bottom = top + hole_size[0]
        right = left + hole_size[1]
        height, weight = img.shape[:2]
        if top < 0 or bottom > height or left < 0 or right > weight:
            raise ValueError(f'crop area {(left, top, right, bottom)} exceeds '
                             f'image size {(height, weight)}')
        img[top:bottom, left:right] = 0
        return img

    def transform(self, results: dict) -> dict:
        """Transform function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        alpha = results['alpha']
        trimap = results['trimap']

        # generate segmentation mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (self.kernel_size,
                                            self.kernel_size))
        seg = (alpha > 0.5).astype(np.float32)
        seg = cv2.erode(
            seg, kernel, iterations=np.random.randint(*self.erode_iter_range))
        seg = cv2.dilate(
            seg, kernel, iterations=np.random.randint(*self.dilate_iter_range))

        # generate some holes in segmentation mask
        num_holes = np.random.randint(*self.num_holes_range)
        for _ in range(num_holes):
            hole_size = random.choice(self.hole_sizes)
            unknown = trimap == 128
            start_point = random_choose_unknown(unknown, hole_size)
            seg = self._crop_hole(seg, start_point, hole_size)
            trimap = self._crop_hole(trimap, start_point, hole_size)

        # perform gaussian blur to segmentation mask
        seg = cv2.GaussianBlur(seg, random.choice(self.blur_ksizes), 0)

        results['seg'] = seg.astype(np.uint8)
        results['num_holes'] = num_holes
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            f'(kernel_size={self.kernel_size}, '
            f'erode_iter_range={self.erode_iter_range}, '
            f'dilate_iter_range={self.dilate_iter_range}, '
            f'num_holes_range={self.num_holes_range}, '
            f'hole_sizes={self.hole_sizes}, blur_ksizes={self.blur_ksizes}')
        return repr_str


@TRANSFORMS.register_module()
class GenerateSoftSeg(BaseTransform):
    """Generate soft segmentation mask from input segmentation mask.

    Required key is "seg", added key is "soft_seg".

    Args:
        fg_thr (float, optional): Threshold of the foreground in the normalized
            input segmentation mask. Defaults to 0.2.
        border_width (int, optional): Width of border to be padded to the
            bottom of the mask. Defaults to 25.
        erode_ksize (int, optional): Fixed kernel size of the erosion.
            Defaults to 5.
        dilate_ksize (int, optional): Fixed kernel size of the dilation.
            Defaults to 5.
        erode_iter_range (tuple, optional): Iteration of erosion.
            Defaults to (10, 20).
        dilate_iter_range (tuple, optional): Iteration of dilation.
            Defaults to (3, 7).
        blur_ksizes (list, optional): List of (h, w) to be selected as the
            kernel_size of the gaussian blur.
            Defaults to [(21, 21), (31, 31), (41, 41)].
    """

    def __init__(self,
                 fg_thr=0.2,
                 border_width=25,
                 erode_ksize=3,
                 dilate_ksize=5,
                 erode_iter_range=(10, 20),
                 dilate_iter_range=(3, 7),
                 blur_ksizes=[(21, 21), (31, 31), (41, 41)]):
        if not isinstance(fg_thr, float):
            raise TypeError(f'fg_thr must be a float, but got {type(fg_thr)}')
        if not isinstance(border_width, int):
            raise TypeError(
                f'border_width must be an int, but got {type(border_width)}')
        if not isinstance(erode_ksize, int):
            raise TypeError(
                f'erode_ksize must be an int, but got {type(erode_ksize)}')
        if not isinstance(dilate_ksize, int):
            raise TypeError(
                f'dilate_ksize must be an int, but got {type(dilate_ksize)}')
        if (not is_tuple_of(erode_iter_range, int)
                or len(erode_iter_range) != 2):
            raise TypeError('erode_iter_range must be a tuple of 2 int, '
                            f'but got {erode_iter_range}')
        if (not is_tuple_of(dilate_iter_range, int)
                or len(dilate_iter_range) != 2):
            raise TypeError('dilate_iter_range must be a tuple of 2 int, '
                            f'but got {dilate_iter_range}')
        if not is_list_of(blur_ksizes, tuple):
            raise TypeError(
                f'blur_ksizes must be a list of tuple, but got {blur_ksizes}')

        self.fg_thr = fg_thr
        self.border_width = border_width
        self.erode_ksize = erode_ksize
        self.dilate_ksize = dilate_ksize
        self.erode_iter_range = erode_iter_range
        self.dilate_iter_range = dilate_iter_range
        self.blur_ksizes = blur_ksizes

    def transform(self, results: dict) -> dict:
        """Transform function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        seg = results['seg'].astype(np.float32) / 255
        height, _ = seg.shape[:2]
        seg[seg > self.fg_thr] = 1

        # to align with the original repo, pad the bottom of the mask
        seg = cv2.copyMakeBorder(seg, 0, self.border_width, 0, 0,
                                 cv2.BORDER_REPLICATE)

        # erode/dilate segmentation mask
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                 (self.erode_ksize,
                                                  self.erode_ksize))
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                  (self.dilate_ksize,
                                                   self.dilate_ksize))
        seg = cv2.erode(
            seg,
            erode_kernel,
            iterations=np.random.randint(*self.erode_iter_range))
        seg = cv2.dilate(
            seg,
            dilate_kernel,
            iterations=np.random.randint(*self.dilate_iter_range))

        # perform gaussian blur to segmentation mask
        seg = cv2.GaussianBlur(seg, random.choice(self.blur_ksizes), 0)

        # remove the padded rows
        seg = (seg * 255).astype(np.uint8)
        seg = np.delete(seg, range(height, height + self.border_width), 0)

        results['soft_seg'] = seg
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(fg_thr={self.fg_thr}, '
                     f'border_width={self.border_width}, '
                     f'erode_ksize={self.erode_ksize}, '
                     f'dilate_ksize={self.dilate_ksize}, '
                     f'erode_iter_range={self.erode_iter_range}, '
                     f'dilate_iter_range={self.dilate_iter_range}, '
                     f'blur_ksizes={self.blur_ksizes})')
        return repr_str
