# Copyright (c) OpenMMLab. All rights reserved.
"""Augmentation on foreground and background."""

import numbers
import os.path as osp

import mmcv
import numpy as np
from mmcv.transforms import BaseTransform
from mmengine.fileio import get_file_backend

from mmagic.registry import TRANSFORMS
from mmagic.utils import add_gaussian_noise, adjust_gamma


@TRANSFORMS.register_module()
class CompositeFg(BaseTransform):
    """Composite foreground with a random foreground.

    This class composites the current training sample with additional data
    randomly (could be from the same dataset). With probability 0.5, the sample
    will be composited with a random sample from the specified directory.
    The composition is performed as:

    .. math::
        fg_{new} = \\alpha_1 * fg_1 + (1 - \\alpha_1) * fg_2

        \\alpha_{new} = 1 - (1 - \\alpha_1) * (1 - \\alpha_2)

    where :math:`(fg_1, \\alpha_1)` is from the current sample and
    :math:`(fg_2, \\alpha_2)` is the randomly loaded sample. With the above
    composition, :math:`\\alpha_{new}` is still in `[0, 1]`.

    Required keys are "alpha" and "fg". Modified keys are "alpha" and "fg".

    Args:
        fg_dirs (str | list[str]): Path of directories to load foreground
            images from.
        alpha_dirs (str | list[str]): Path of directories to load alpha mattes
            from.
        interpolation (str): Interpolation method of `mmcv.imresize` to resize
            the randomly loaded images. Default: 'nearest'.
    """

    def __init__(self, fg_dirs, alpha_dirs, interpolation='nearest'):
        # TODO try fetch the path from dataset
        self.fg_dirs = fg_dirs if isinstance(fg_dirs, list) else [fg_dirs]
        self.alpha_dirs = alpha_dirs if isinstance(alpha_dirs,
                                                   list) else [alpha_dirs]
        self.interpolation = interpolation

        self.file_backend = get_file_backend(uri=fg_dirs[0])

        self.fg_list, self.alpha_list = self._get_file_list(
            self.fg_dirs, self.alpha_dirs)

    def transform(self, results: dict) -> dict:
        """Transform function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        fg = results['fg']
        alpha = results['alpha'] / 255.0  # float64, H, W, 1
        h, w = results['fg'].shape[:2]

        # randomly select fg
        if np.random.rand() < 0.5:
            idx = np.random.randint(len(self.fg_list))
            fg2_bytes = self.file_backend.get(self.fg_list[idx])
            fg2 = mmcv.imfrombytes(fg2_bytes)
            alpha2_bytes = self.file_backend.get(self.alpha_list[idx])
            alpha2 = mmcv.imfrombytes(alpha2_bytes, flag='grayscale')
            alpha2 = alpha2 / 255.0  # float64
            fg2 = mmcv.imresize(fg2, (w, h), interpolation=self.interpolation)
            alpha2 = mmcv.imresize(
                alpha2, (w, h), interpolation=self.interpolation)
            alpha2 = alpha2[..., None]

            # the overlap of two 50% transparency will be 75%
            alpha_tmp = 1 - (1 - alpha) * (1 - alpha2)
            # if the result alpha is all-one, then we avoid composition
            if np.any(alpha_tmp < 1):
                # composite fg with fg2
                fg = fg * alpha + fg2 * (1 - alpha)
                alpha = alpha_tmp

        results['fg'] = fg
        results['alpha'] = alpha * 255
        return results

    def _get_file_list(self, fg_dirs, alpha_dirs):
        all_fg_list = list()
        all_alpha_list = list()
        for fg_dir, alpha_dir in zip(fg_dirs, alpha_dirs):
            fg_list = sorted(
                self.file_backend.list_dir_or_file(fg_dir, list_dir=False))
            alpha_list = sorted(
                self.file_backend.list_dir_or_file(alpha_dir, list_dir=False))
            # we assume the file names for fg and alpha are the same
            assert len(fg_list) == len(alpha_list), (
                f'{fg_dir} and {alpha_dir} should have the same number of '
                f'images ({len(fg_list)} differs from ({len(alpha_list)})')
            fg_list = [osp.join(fg_dir, fg) for fg in fg_list]
            alpha_list = [osp.join(alpha_dir, alpha) for alpha in alpha_list]

            all_fg_list.extend(fg_list)
            all_alpha_list.extend(alpha_list)
        return all_fg_list, all_alpha_list

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(fg_dirs={repr(self.fg_dirs)}, '
                     f'alpha_dirs={repr(self.alpha_dirs)}, '
                     f'interpolation={repr(self.interpolation)})')
        return repr_str


@TRANSFORMS.register_module()
class MergeFgAndBg(BaseTransform):
    """Composite foreground image and background image with alpha.

    Required keys are "alpha", "fg" and "bg", added key is "merged".
    """

    def transform(self, results: dict) -> dict:
        """Transform function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        alpha = results['alpha'].astype(np.float32) / 255.
        fg = results['fg']
        bg = results['bg']
        merged = fg * alpha + (1. - alpha) * bg
        results['merged'] = merged
        return results

    def __repr__(self) -> str:
        repr_str = f'{self.__class__.__name__}()'
        return repr_str


@TRANSFORMS.register_module()
class PerturbBg(BaseTransform):
    """Randomly add gaussian noise or gamma change to background image.

    Required key is "bg", added key is "noisy_bg".

    Args:
        gamma_ratio (float, optional): The probability to use gamma correction
            instead of gaussian noise. Defaults to 0.6.
    """

    def __init__(self, gamma_ratio=0.6):
        if gamma_ratio < 0 or gamma_ratio > 1:
            raise ValueError('gamma_ratio must be a float between [0, 1], '
                             f'but got {gamma_ratio}')
        self.gamma_ratio = gamma_ratio

    def transform(self, results: dict) -> dict:
        """Transform function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        if np.random.rand() >= self.gamma_ratio:
            # generate gaussian noise with random gaussian N([-7, 7), [2, 6))
            mu = np.random.randint(-7, 7)
            sigma = np.random.randint(2, 6)
            results['noisy_bg'] = add_gaussian_noise(results['bg'], mu, sigma)
        else:
            # adjust gamma in a range of N(1, 0.12)
            gamma = np.random.normal(1, 0.12)
            results['noisy_bg'] = adjust_gamma(results['bg'], gamma)
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(gamma_ratio={self.gamma_ratio})'


@TRANSFORMS.register_module()
class RandomJitter(BaseTransform):
    """Randomly jitter the foreground in hsv space.

    The jitter range of hue is adjustable while the jitter ranges of saturation
    and value are adaptive to the images. Side effect: the "fg" image will be
    converted to `np.float32`.
    Required keys are "fg" and "alpha", modified key is "fg".

    Args:
        hue_range (float | tuple[float]): Range of hue jittering. If it is a
            float instead of a tuple like (min, max), the range of hue
            jittering will be (-hue_range, +hue_range). Default: 40.
    """

    def __init__(self, hue_range=40):

        if isinstance(hue_range, numbers.Number):
            assert hue_range >= 0, ('If hue_range is a single number, '
                                    'it must be positive.')
            self.hue_range = (-hue_range, hue_range)
        else:
            assert isinstance(hue_range, tuple) and len(hue_range) == 2, \
                'hue_range should be a tuple and it must be of length 2.'
            self.hue_range = hue_range

    def transform(self, results):
        """transform function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """

        fg, alpha = results['fg'], results['alpha']
        alpha = alpha[:, :, 0]

        # convert to HSV space;
        # convert to float32 image to keep precision during space conversion.
        fg = mmcv.bgr2hsv(fg.astype(np.float32) / 255)
        # Hue noise
        hue_jitter = np.random.randint(self.hue_range[0], self.hue_range[1])
        fg[:, :, 0] = np.remainder(fg[:, :, 0] + hue_jitter, 360)

        # Saturation noise
        sat_mean = fg[:, :, 1][alpha > 0].mean()
        # jitter saturation within range (1.1 - sat_mean) * [-0.1, 0.1]
        sat_jitter = (1.1 - sat_mean) * (np.random.rand() * 0.2 - 0.1)
        sat = fg[:, :, 1]
        sat = np.abs(sat + sat_jitter)
        sat[sat > 1] = 2 - sat[sat > 1]
        fg[:, :, 1] = sat

        # Value noise
        val_mean = fg[:, :, 2][alpha > 0].mean()
        # jitter value within range (1.1 - val_mean) * [-0.1, 0.1]
        val_jitter = (1.1 - val_mean) * (np.random.rand() * 0.2 - 0.1)
        val = fg[:, :, 2]
        val = np.abs(val + val_jitter)
        val[val > 1] = 2 - val[val > 1]
        fg[:, :, 2] = val
        # convert back to BGR space
        fg = mmcv.hsv2bgr(fg)
        results['fg'] = fg * 255

        return results

    def __repr__(self):

        return self.__class__.__name__ + f'hue_range={self.hue_range}'


@TRANSFORMS.register_module()
class RandomLoadResizeBg(BaseTransform):
    """Randomly load a background image and resize it.

    Required key is "fg", added key is "bg".

    Args:
        bg_dir (str): Path of directory to load background images from.
        flag (str): Loading flag for images. Default: 'color'.
        channel_order (str): Order of channel, candidates are 'bgr' and 'rgb'.
            Default: 'bgr'.
        kwargs (dict): Args for file client.
    """

    def __init__(self, bg_dir, flag='color', channel_order='bgr'):
        self.bg_dir = bg_dir

        self.file_backend = get_file_backend(uri=bg_dir)
        self.bg_list = list(
            self.file_backend.list_dir_or_file(bg_dir, list_dir=False))

        self.flag = flag
        self.channel_order = channel_order

    def transform(self, results: dict) -> dict:
        """Transform function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        h, w = results['fg'].shape[:2]
        idx = np.random.randint(len(self.bg_list))
        filepath = f'{self.bg_dir}/{self.bg_list[idx]}'
        img_bytes = self.file_backend.get(filepath)
        img = mmcv.imfrombytes(
            img_bytes, flag=self.flag, channel_order=self.channel_order)  # HWC
        bg = mmcv.imresize(img, (w, h), interpolation='bicubic')
        results['bg'] = bg
        return results

    def __repr__(self):
        return self.__class__.__name__ + f"(bg_dir='{self.bg_dir}')"
