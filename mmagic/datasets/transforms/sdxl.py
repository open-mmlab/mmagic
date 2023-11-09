# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Sequence, Union

import mmcv
import numpy as np
from mmcv.transforms import BaseTransform, to_tensor

from mmagic.datasets.transforms.aug_shape import Flip
from mmagic.registry import TRANSFORMS


@TRANSFORMS.register_module()
class RandomCropXL(BaseTransform):
    """Random crop the given image. Required Keys:

    - [KEYS]

    Modified Keys:
    - [KEYS]

    New Keys:
    - [KEYS]_crop_bbox

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted
            as (size[0], size[0])
        keys (str or list[str]): The images to be cropped.
    """

    def __init__(self, size: int, keys: Union[str, List[str]] = 'img'):
        if not isinstance(size, Sequence):
            size = (size, size)
        self.size = size

        assert keys, 'Keys should not be empty.'
        if not isinstance(keys, list):
            keys = [keys]
        self.keys = keys

    def transform(self, results: Dict) -> Dict:
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        assert all(results[self.keys[0]].size == results[k].size
                   for k in self.keys)

        h, w, _ = results[self.keys[0]].shape

        if h < self.size[0] or w < self.size[1]:
            raise ValueError(
                f'({h}, {w}) is smaller than crop size {self.size}.')

        # randomly choose top and left coordinates for img patch
        top = np.random.randint(h - self.size[0] + 1)
        left = np.random.randint(w - self.size[1] + 1)

        for key in self.keys:
            results[key] = results[key][top:top + self.size[0],
                                        left:left + self.size[1], ...]
            results[f'{key}_crop_bbox'] = [
                top, left, top + self.size[0], left + self.size[1]
            ]

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(size={self.size}, ' f'keys={self.keys})')
        return repr_str


@TRANSFORMS.register_module()
class FlipXL(Flip):
    """Flip the input data with a probability.

    The differences between FlipXL & Flip:
        1. Fix [KEYS]_crop_bbox.

    Required Keys:
    - [KEYS]
    - [KEYS]_crop_bbox

    Modified Keys:
    - [KEYS]
    - [KEYS]_crop_bbox

    Args:
        keys (Union[str, List[str]]): The images to be flipped.
        flip_ratio (float): The probability to flip the images. Default: 0.5.
        direction (str): Flip images horizontally or vertically. Options are
            "horizontal" | "vertical". Default: "horizontal".
    """

    def transform(self, results: Dict) -> Dict:
        """transform function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """

        flip = np.random.random() < self.flip_ratio

        if flip:
            for key in self.keys:
                mmcv.imflip_(results[key], self.direction)
                h, w, _ = results[key].shape
                if self.direction == 'horizontal':
                    results[f'{key}_crop_bbox'] = [
                        results[f'{key}_crop_bbox'][0],
                        w - results[f'{key}_crop_bbox'][3],
                        results[f'{key}_crop_bbox'][2],
                        w - results[f'{key}_crop_bbox'][1]
                    ]
                elif self.direction == 'vertical':
                    results[f'{key}_crop_bbox'] = [
                        h - results[f'{key}_crop_bbox'][2],
                        results[f'{key}_crop_bbox'][1],
                        h - results[f'{key}_crop_bbox'][0],
                        results[f'{key}_crop_bbox'][3]
                    ]

        if 'flip_infos' not in results:
            results['flip_infos'] = []

        flip_info = dict(
            keys=self.keys,
            direction=self.direction,
            ratio=self.flip_ratio,
            flip=flip)
        results['flip_infos'].append(flip_info)

        return results


@TRANSFORMS.register_module()
class ComputeTimeIds(BaseTransform):
    """Load a single image from corresponding paths. Required Required Keys:

    - [Key]
    - ori_[KEY]_shape
    - [KEYS]_crop_bbox

    New Keys:
    - time_ids

    Args:
        key (str): Keys in results to find corresponding path.
            Defaults to `img`.
    """

    def __init__(
        self,
        key: str = 'img',
    ) -> None:

        self.key = key

    def transform(self, results: Dict) -> Dict:
        """
        Args:
            results (dict): The result dict.

        Returns:
            dict: 'time_ids' key is added as original image shape.
        """
        assert f'ori_{self.key}_shape' in results
        assert f'{self.key}_crop_bbox' in results
        target_size = list(results[self.key].shape)[:2]
        time_ids = list(results[f'ori_{self.key}_shape'][:2]
                        ) + results[f'{self.key}_crop_bbox'][:2] + target_size
        results['time_ids'] = to_tensor(time_ids)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(key={self.key})')
        return repr_str


@TRANSFORMS.register_module()
class ResizeEdge(BaseTransform):
    """Resize images along the specified edge.

    Required Keys:
    - [KEYS]

    Modified Keys:
    - [KEYS]
    - [KEYS]_shape

    New Keys:
    - keep_ratio
    - scale_factor
    - interpolation

    Args:
        scale (int): The edge scale to resizing.
        keys (str | list[str]): The image(s) to be resized.
        edge (str): The edge to resize. Defaults to 'short'.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results.
            Defaults to 'cv2'.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend.
            Defaults to 'bilinear'.
    """

    def __init__(self,
                 scale: int,
                 keys: Union[str, List[str]] = 'img',
                 edge: str = 'short',
                 backend: str = 'cv2',
                 interpolation: str = 'bilinear') -> None:
        assert keys, 'Keys should not be empty.'
        keys = [keys] if not isinstance(keys, list) else keys

        allow_edges = ['short', 'long', 'width', 'height']
        assert edge in allow_edges, \
            f'Invalid edge "{edge}", please specify from {allow_edges}.'

        self.keys = keys
        self.edge = edge
        self.scale = scale
        self.backend = backend
        self.interpolation = interpolation

    def _resize_img(self, results: dict, key: str) -> None:
        """Resize images with ``results['scale']``."""

        img, w_scale, h_scale = mmcv.imresize(
            results[key],
            results['scale'],
            interpolation=self.interpolation,
            return_scale=True,
            backend=self.backend)
        results[key] = img
        results[f'{key}_shape'] = img.shape[:2]
        results['scale_factor'] = (w_scale, h_scale)
        results['keep_ratio'] = True
        results['interpolation'] = self.interpolation

    def transform(self, results: Dict) -> Dict:
        """Transform function to resize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img', 'scale', 'scale_factor',
            'img_shape' keys are updated in result dict.
        """
        for k in self.keys:
            assert k in results, f'No {k} field in the input.'

            h, w = results[k].shape[:2]
            if any([
                    # conditions to resize the width
                    self.edge == 'short' and w < h,
                    self.edge == 'long' and w > h,
                    self.edge == 'width',
            ]):
                width = self.scale
                height = int(self.scale * h / w)
            else:
                height = self.scale
                width = int(self.scale * w / h)
            results['scale'] = (width, height)

            self._resize_img(results, k)
        return results

    def __repr__(self):
        """Print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f'(scale={self.scale}, '
        repr_str += f'edge={self.edge}, '
        repr_str += f'backend={self.backend}, '
        repr_str += f'interpolation={self.interpolation})'
        return repr_str
