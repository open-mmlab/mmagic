# Copyright (c) OpenMMLab. All rights reserved.
"""Crop used in matting augmentations."""
import mmcv
import numpy as np
from mmcv.transforms import BaseTransform
from torch.nn.modules.utils import _pair

from mmedit.registry import TRANSFORMS
from .utils import random_choose_unknown


@TRANSFORMS.register_module()
class CropAroundCenter(BaseTransform):
    """Randomly crop the images around unknown area in the center 1/4 images.

    This cropping strategy is adopted in GCA matting. The `unknown area` is the
    same as `semi-transparent area`.
    https://arxiv.org/pdf/2001.04069.pdf

    It retains the center 1/4 images and resizes the images to 'crop_size'.
    Required keys are "fg", "bg", "trimap" and "alpha", added or modified keys
    are "crop_bbox", "fg", "bg", "trimap" and "alpha".

    Args:
        crop_size (int | tuple): Desired output size. If int, square crop is
            applied.
    """

    def __init__(self, crop_size):
        if mmcv.is_tuple_of(crop_size, int):
            assert len(crop_size) == 2, 'length of crop_size must be 2.'
        elif not isinstance(crop_size, int):
            raise TypeError('crop_size must be int or a tuple of int, but got '
                            f'{type(crop_size)}')
        self.crop_size = _pair(crop_size)

    def transform(self, results):
        """Transform function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """

        fg = results['fg']
        alpha = results['alpha']
        trimap = results['trimap']
        bg = results['bg']
        h, w = fg.shape[:2]
        assert bg.shape == fg.shape, (f'shape of bg {bg.shape} should be the '
                                      f'same as fg {fg.shape}.')

        crop_h, crop_w = self.crop_size
        # Make sure h >= crop_h, w >= crop_w. If not, rescale imgs
        rescale_ratio = max(crop_h / h, crop_w / w)
        if rescale_ratio > 1:
            new_h = max(int(h * rescale_ratio), crop_h)
            new_w = max(int(w * rescale_ratio), crop_w)
            fg = mmcv.imresize(fg, (new_w, new_h), interpolation='nearest')
            alpha = mmcv.imresize(
                alpha, (new_w, new_h), interpolation='nearest')
            trimap = mmcv.imresize(
                trimap, (new_w, new_h), interpolation='nearest')
            bg = mmcv.imresize(bg, (new_w, new_h), interpolation='bicubic')
            h, w = new_h, new_w

        # resize to 1/4 to ignore small unknown patches
        small_trimap = mmcv.imresize(
            trimap, (w // 4, h // 4), interpolation='nearest')
        # find unknown area in center 1/4 region
        margin_h, margin_w = crop_h // 2, crop_w // 2
        sample_area = small_trimap[margin_h // 4:(h - margin_h) // 4,
                                   margin_w // 4:(w - margin_w) // 4]
        unknown_xs, unknown_ys = np.where(sample_area == 128)
        unknown_num = len(unknown_xs)
        if unknown_num < 10:
            # too few unknown area in the center, crop from the whole image
            top = np.random.randint(0, h - crop_h + 1)
            left = np.random.randint(0, w - crop_w + 1)
        else:
            idx = np.random.randint(unknown_num)
            top = unknown_xs[idx] * 4
            left = unknown_ys[idx] * 4
        bottom = top + crop_h
        right = left + crop_w

        results['fg'] = fg[top:bottom, left:right]
        results['alpha'] = alpha[top:bottom, left:right]
        results['trimap'] = trimap[top:bottom, left:right]
        results['bg'] = bg[top:bottom, left:right]
        results['crop_bbox'] = (left, top, right, bottom)

        return results

    def __repr__(self):

        return self.__class__.__name__ + f'(crop_size={self.crop_size})'


@TRANSFORMS.register_module()
class CropAroundFg(BaseTransform):
    """Crop around the whole foreground in the segmentation mask.

    Required keys are "seg" and the keys in argument `keys`.
    Meanwhile, "seg" must be in argument `keys`. Added or modified keys are
    "crop_bbox" and the keys in argument `keys`.

    Args:
        keys (Sequence[str]): The images to be cropped. It must contain
            'seg'.
        bd_ratio_range (tuple, optional): The range of the boundary (bd) ratio
            to select from. The boundary ratio is the ratio of the boundary to
            the minimal bbox that contains the whole foreground given by
            segmentation. Default to (0.1, 0.4).
        test_mode (bool): Whether use test mode. In test mode, the tight crop
            area of foreground will be extended to the a square.
            Default to False.
    """

    def __init__(self, keys, bd_ratio_range=(0.1, 0.4), test_mode=False):

        if 'seg' not in keys:
            raise ValueError(f'"seg" must be in keys, but got {keys}')
        if (not mmcv.is_tuple_of(bd_ratio_range, float)
                or len(bd_ratio_range) != 2):
            raise TypeError('bd_ratio_range must be a tuple of 2 int, but got '
                            f'{bd_ratio_range}')
        self.keys = keys
        self.bd_ratio_range = bd_ratio_range
        self.test_mode = test_mode

    def transform(self, results):
        """Transform function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """

        seg = results['seg']
        height, width = seg.shape[:2]

        # get foreground bbox
        fg_coor = np.array(np.where(seg))
        top, left = np.amin(fg_coor, axis=1)
        bottom, right = np.amax(fg_coor, axis=1)

        # enlarge bbox
        long_side = np.maximum(bottom - top, right - left)
        if self.test_mode:
            bottom = top + long_side
            right = left + long_side
        boundary_ratio = np.random.uniform(*self.bd_ratio_range)
        boundary = int(np.round(boundary_ratio * long_side))
        # NOTE: Different from the original repo, we keep track of the four
        # corners of the bbox (left, top, right, bottom) while the original
        # repo use (top, left, height, width) to represent bbox. This may
        # introduce an difference of 1 pixel.
        top = max(top - boundary, 0)
        left = max(left - boundary, 0)
        bottom = min(bottom + boundary, height)
        right = min(right + boundary, width)

        for key in self.keys:
            results[key] = results[key][top:bottom, left:right]
        results['crop_bbox'] = (left, top, right, bottom)

        return results


@TRANSFORMS.register_module()
class CropAroundUnknown(BaseTransform):
    """Crop around unknown area with a randomly selected scale.

    Randomly select the w and h from a list of (w, h).
    Required keys are the keys in argument `keys`, added or
    modified keys are "crop_bbox" and the keys in argument `keys`.
    This class assumes value of "alpha" ranges from 0 to 255.

    Args:
        keys (Sequence[str]): The images to be cropped. It must contain
            'alpha'. If unknown_source is set to 'trimap', then it must also
            contain 'trimap'.
        crop_sizes (list[int | tuple[int]]): List of (w, h) to be selected.
        unknown_source (str, optional): Unknown area to select from. It must be
            'alpha' or 'tirmap'. Default to 'alpha'.
        interpolations (str | list[str], optional): Interpolation method of
            mmcv.imresize. The interpolation operation will be applied when
            image size is smaller than the crop_size. If given as a list of
            str, it should have the same length as `keys`. Or if given as a
            str all the keys will be resized with the same method.
            Default to 'bilinear'.
    """

    def __init__(self,
                 keys,
                 crop_sizes,
                 unknown_source='alpha',
                 interpolations='bilinear'):
        if 'alpha' not in keys:
            raise ValueError(f'"alpha" must be in keys, but got {keys}')
        self.keys = keys

        if not isinstance(crop_sizes, list):
            raise TypeError(
                f'Crop sizes must be list, but got {type(crop_sizes)}.')
        self.crop_sizes = [_pair(crop_size) for crop_size in crop_sizes]
        if not mmcv.is_tuple_of(self.crop_sizes[0], int):
            raise TypeError('Elements of crop_sizes must be int or tuple of '
                            f'int, but got {type(self.crop_sizes[0][0])}.')

        if unknown_source not in ['alpha', 'trimap']:
            raise ValueError('unknown_source must be "alpha" or "trimap", '
                             f'but got {unknown_source}')
        if unknown_source not in keys:
            # it could only be trimap, since alpha is checked before
            raise ValueError(
                'if unknown_source is "trimap", it must also be set in keys')
        self.unknown_source = unknown_source

        if isinstance(interpolations, str):
            self.interpolations = [interpolations] * len(self.keys)
        elif mmcv.is_list_of(interpolations,
                             str) and len(interpolations) == len(self.keys):
            self.interpolations = interpolations
        else:
            raise TypeError(
                'interpolations must be a str or list of str with '
                f'the same length as keys, but got {interpolations}')

    def transform(self, results):
        """Transform function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """

        h, w = results[self.keys[0]].shape[:2]

        rand_ind = np.random.randint(len(self.crop_sizes))
        crop_h, crop_w = self.crop_sizes[rand_ind]

        # Make sure h >= crop_h, w >= crop_w. If not, rescale imgs
        rescale_ratio = max(crop_h / h, crop_w / w)
        if rescale_ratio > 1:
            h = max(int(h * rescale_ratio), crop_h)
            w = max(int(w * rescale_ratio), crop_w)
            for key, interpolation in zip(self.keys, self.interpolations):
                results[key] = mmcv.imresize(
                    results[key], (w, h), interpolation=interpolation)

        # Select the cropping top-left point which is an unknown pixel
        if self.unknown_source == 'alpha':
            unknown = (results['alpha'] > 0) & (results['alpha'] < 255)
        else:
            unknown = results['trimap'] == 128
        top, left = random_choose_unknown(unknown.squeeze(), (crop_h, crop_w))

        bottom = top + crop_h
        right = left + crop_w

        for key in self.keys:
            results[key] = results[key][top:bottom, left:right]
        results['crop_bbox'] = (left, top, right, bottom)

        return results

    def __repr__(self):

        repr_str = self.__class__.__name__
        repr_str += (f'(keys={self.keys}, crop_sizes={self.crop_sizes}, '
                     f"unknown_source='{self.unknown_source}', "
                     f'interpolations={self.interpolations})')

        return repr_str
