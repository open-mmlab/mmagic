# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
from mmcv.transforms import BaseTransform

from mmedit.registry import TRANSFORMS


@TRANSFORMS.register_module()
class Normalize(BaseTransform):
    """Normalize images with the given mean and std value.

    Required keys are the keys in attribute "keys", added or modified keys are
    the keys in attribute "keys" and these keys with postfix '_norm_cfg'.
    It also supports normalizing a list of images.

    Args:
        keys (Sequence[str]): The images to be normalized.
        mean (np.ndarray): Mean values of different channels.
        std (np.ndarray): Std values of different channels.
        to_rgb (bool): Whether to convert channels from BGR to RGB.
            Default: False.
        save_original (bool): Whether to save original images. Default: False.
    """

    def __init__(self, keys, mean, std, to_rgb=False, save_original=False):
        self.keys = keys
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb
        self.save_original = save_original

    def transform(self, results):
        """transform function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        for key in self.keys:
            if isinstance(results[key], list):
                if self.save_original:
                    results[key + '_unnormalised'] = [
                        v.copy() for v in results[key]
                    ]
                results[key] = [
                    mmcv.imnormalize(v, self.mean, self.std, self.to_rgb)
                    for v in results[key]
                ]
            else:
                if self.save_original:
                    results[key + '_unnormalised'] = results[key].copy()
                results[key] = mmcv.imnormalize(results[key], self.mean,
                                                self.std, self.to_rgb)

        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(keys={self.keys}, mean={self.mean}, std={self.std}, '
                     f'to_rgb={self.to_rgb})')

        return repr_str


@TRANSFORMS.register_module()
class RescaleToZeroOne(BaseTransform):
    """Transform the images into a range between 0 and 1.

    Required keys are the keys in attribute "keys", added or modified keys are
    the keys in attribute "keys".
    It also supports rescaling a list of images.

    Args:
        keys (Sequence[str]): The images to be transformed.
    """

    def __init__(self, keys):
        self.keys = keys

    def transform(self, results):
        """transform function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        for key in self.keys:
            if isinstance(results[key], list):
                results[key] = [
                    v.astype(np.float32) / 255. for v in results[key]
                ]
            else:
                results[key] = results[key].astype(np.float32) / 255.
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'
