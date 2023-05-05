# Copyright (c) OpenMMLab. All rights reserved.
import random
from copy import deepcopy
from typing import Dict, List, Union

import mmcv
import numpy as np
from mmcv.transforms import BaseTransform
from mmengine.utils import is_tuple_of

from mmagic.registry import TRANSFORMS


@TRANSFORMS.register_module()
class Flip(BaseTransform):
    """Flip the input data with a probability.

    Reverse the order of elements in the given data with a specific direction.
    The shape of the data is preserved, but the elements are reordered.
    Required keys are the keys in attributes "keys", added or modified keys are
    "flip", "flip_direction" and the keys in attributes "keys".
    It also supports flipping a list of images with the same flip.

    Required Keys:

    - [KEYS]

    Modified Keys:

    - [KEYS]

    Args:
        keys (Union[str, List[str]]): The images to be flipped.
        flip_ratio (float): The probability to flip the images. Default: 0.5.
        direction (str): Flip images horizontally or vertically. Options are
            "horizontal" | "vertical". Default: "horizontal".
    """
    _directions = ['horizontal', 'vertical']

    def __init__(self, keys, flip_ratio=0.5, direction='horizontal'):

        if direction not in self._directions:
            raise ValueError(f'Direction {direction} is not supported.'
                             f'Currently support ones are {self._directions}')

        self.keys = keys if isinstance(keys, list) else [keys]
        self.flip_ratio = flip_ratio
        self.direction = direction

    def transform(self, results):
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
                if isinstance(results[key], list):
                    for v in results[key]:
                        mmcv.imflip_(v, self.direction)
                else:
                    mmcv.imflip_(results[key], self.direction)

        if 'flip_infos' not in results:
            results['flip_infos'] = []

        flip_info = dict(
            keys=self.keys,
            direction=self.direction,
            ratio=self.flip_ratio,
            flip=flip)
        results['flip_infos'].append(flip_info)

        return results

    def __repr__(self):

        repr_str = self.__class__.__name__
        repr_str += (f'(keys={self.keys}, flip_ratio={self.flip_ratio}, '
                     f'direction={self.direction})')

        return repr_str


@TRANSFORMS.register_module()
class RandomRotation(BaseTransform):
    """Rotate the image by a randomly-chosen angle, measured in degree.

    Args:
        keys (list[str]): The images to be rotated.
        degrees (tuple[float] | tuple[int] | float | int): If it is a tuple,
            it represents a range (min, max). If it is a float or int,
            the range is constructed as (-degrees, degrees).
    """

    def __init__(self, keys, degrees):

        if isinstance(degrees, (int, float)):
            if degrees < 0.0:
                raise ValueError('Degrees must be positive if it is a number.')
            else:
                degrees = (-degrees, degrees)
        elif not is_tuple_of(degrees, (int, float)):
            raise TypeError(f'Degrees must be float | int or tuple of float | '
                            'int, but got '
                            f'{type(degrees)}.')

        self.keys = keys
        self.degrees = degrees

    def transform(self, results):
        """transform function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """

        angle = random.uniform(self.degrees[0], self.degrees[1])

        for k in self.keys:
            results[k] = mmcv.imrotate(results[k], angle)
            if results[k].ndim == 2:
                results[k] = np.expand_dims(results[k], axis=2)
        results['degrees'] = self.degrees

        return results

    def __repr__(self):

        repr_str = self.__class__.__name__
        repr_str += (f'(keys={self.keys}, degrees={self.degrees})')

        return repr_str


@TRANSFORMS.register_module()
class RandomTransposeHW(BaseTransform):
    """Randomly transpose images in H and W dimensions with a probability.

    (TransposeHW = horizontal flip + anti-clockwise rotation by 90 degrees)
    When used with horizontal/vertical flips, it serves as a way of rotation
    augmentation.
    It also supports randomly transposing a list of images.

    Required keys are the keys in attributes "keys", added or modified keys are
    "transpose" and the keys in attributes "keys".

    Args:
        keys (list[str]): The images to be transposed.
        transpose_ratio (float): The probability to transpose the images.
            Default: 0.5.
    """

    def __init__(self, keys, transpose_ratio=0.5):

        self.keys = keys if isinstance(keys, list) else [keys]
        self.transpose_ratio = transpose_ratio

    def transform(self, results):
        """transform function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """

        transpose = np.random.random() < self.transpose_ratio

        if transpose:
            for key in self.keys:
                if isinstance(results[key], list):
                    results[key] = [v.transpose(1, 0, 2) for v in results[key]]
                else:
                    results[key] = results[key].transpose(1, 0, 2)

        results['transpose'] = transpose

        return results

    def __repr__(self):

        repr_str = self.__class__.__name__
        repr_str += (
            f'(keys={self.keys}, transpose_ratio={self.transpose_ratio})')

        return repr_str


@TRANSFORMS.register_module()
class Resize(BaseTransform):
    """Resize data to a specific size for training or resize the images to fit
    the network input regulation for testing.

    When used for resizing images to fit network input regulation, the case is
    that a network may have several downsample and then upsample operation,
    then the input height and width should be divisible by the downsample
    factor of the network.
    For example, the network would downsample the input for 5 times with
    stride 2, then the downsample factor is 2^5 = 32 and the height
    and width should be divisible by 32.

    Required keys are the keys in attribute "keys", added or modified keys are
    "keep_ratio", "scale_factor", "interpolation" and the
    keys in attribute "keys".

    Required Keys:

    - Required keys are the keys in attribute "keys"

    Modified Keys:

    - Modified the keys in attribute "keys" or save as new key ([OUT_KEY])

    Added Keys:

    - [OUT_KEY]_shape
    - keep_ratio
    - scale_factor
    - interpolation

    All keys in "keys" should have the same shape. "test_trans" is used to
    record the test transformation to align the input's shape.

    Args:
        keys (str | list[str]): The image(s) to be resized.
        scale (float | tuple[int]): If scale is tuple[int], target spatial
            size (h, w). Otherwise, target spatial size is scaled by input
            size.
            Note that when it is used, `size_factor` and `max_size` are
            useless. Default: None
        keep_ratio (bool): If set to True, images will be resized without
            changing the aspect ratio. Otherwise, it will resize images to a
            given size. Default: False.
            Note that it is used together with `scale`.
        size_factor (int): Let the output shape be a multiple of size_factor.
            Default:None.
            Note that when it is used, `scale` should be set to None and
            `keep_ratio` should be set to False.
        max_size (int): The maximum size of the longest side of the output.
            Default:None.
            Note that it is used together with `size_factor`.
        interpolation (str): Algorithm used for interpolation:
            "nearest" | "bilinear" | "bicubic" | "area" | "lanczos".
            Default: "bilinear".
        backend (str | None): The image resize backend type. Options are `cv2`,
            `pillow`, `None`. If backend is None, the global imread_backend
            specified by ``mmcv.use_backend()`` will be used.
            Default: None.
        output_keys (list[str] | None): The resized images. Default: None
            Note that if it is not `None`, its length should be equal to keys.
    """

    def __init__(self,
                 keys: Union[str, List[str]] = 'img',
                 scale=None,
                 keep_ratio=False,
                 size_factor=None,
                 max_size=None,
                 interpolation='bilinear',
                 backend=None,
                 output_keys=None):

        assert keys, 'Keys should not be empty.'
        keys = [keys] if not isinstance(keys, list) else keys
        if output_keys:
            assert len(output_keys) == len(keys)
        else:
            output_keys = keys
        if size_factor:
            assert scale is None, ('When size_factor is used, scale should ',
                                   f'be None. But received {scale}.')
            assert keep_ratio is False, ('When size_factor is used, '
                                         'keep_ratio should be False.')
        if max_size:
            assert size_factor is not None, (
                'When max_size is used, '
                f'size_factor should also be set. But received {size_factor}.')
        if isinstance(scale, float):
            if scale <= 0:
                raise ValueError(f'Invalid scale {scale}, must be positive.')
        elif is_tuple_of(scale, int):
            max_long_edge = max(scale)
            max_short_edge = min(scale)
            if max_short_edge == -1:
                # assign np.inf to long edge for rescaling short edge later.
                scale = (np.inf, max_long_edge)
        elif scale is not None:
            raise TypeError(
                f'Scale must be None, float or tuple of int, but got '
                f'{type(scale)}.')

        self.keys = keys
        self.output_keys = output_keys
        self.scale = scale
        self.size_factor = size_factor
        self.max_size = max_size
        self.keep_ratio = keep_ratio
        self.interpolation = interpolation
        self.backend = backend

    def _resize(self, img):
        """Resize function.

        Args:
            img (np.ndarray): Image.

        Returns:
            img (np.ndarray): Resized image.
        """
        if isinstance(img, list):
            for i, image in enumerate(img):
                size, img[i] = self._resize(image)
            return size, img
        else:
            if self.keep_ratio:
                img, self.scale_factor = mmcv.imrescale(
                    img,
                    self.scale,
                    return_scale=True,
                    interpolation=self.interpolation,
                    backend=self.backend)
            else:
                img, w_scale, h_scale = mmcv.imresize(
                    img,
                    self.scale,
                    return_scale=True,
                    interpolation=self.interpolation,
                    backend=self.backend)
                self.scale_factor = np.array((w_scale, h_scale),
                                             dtype=np.float32)

            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=2)
            return img.shape, img

    def transform(self, results: Dict) -> Dict:
        """Transform function to resize images.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        if self.size_factor:
            h, w = results[self.keys[0]].shape[:2]
            new_h = h - (h % self.size_factor)
            new_w = w - (w % self.size_factor)
            if self.max_size:
                new_h = min(self.max_size - (self.max_size % self.size_factor),
                            new_h)
                new_w = min(self.max_size - (self.max_size % self.size_factor),
                            new_w)
            self.scale = (new_w, new_h)

        for key, out_key in zip(self.keys, self.output_keys):
            if key in results:
                size, results[out_key] = self._resize(results[key])
                results[f'{out_key}_shape'] = size
                # copy metainfo
                if f'ori_{key}_shape' in results:
                    results[f'ori_{out_key}_shape'] = deepcopy(
                        results[f'ori_{key}_shape'])
                if f'{key}_channel_order' in results:
                    results[f'{out_key}_channel_order'] = deepcopy(
                        results[f'{key}_channel_order'])
                if f'{key}_color_type' in results:
                    results[f'{out_key}_color_type'] = deepcopy(
                        results[f'{key}_color_type'])

        results['scale_factor'] = self.scale_factor
        results['keep_ratio'] = self.keep_ratio
        results['interpolation'] = self.interpolation
        results['backend'] = self.backend

        return results

    def __repr__(self):

        repr_str = self.__class__.__name__
        repr_str += (
            f'(keys={self.keys}, output_keys={self.output_keys}, '
            f'scale={self.scale}, '
            f'keep_ratio={self.keep_ratio}, size_factor={self.size_factor}, '
            f'max_size={self.max_size}, interpolation={self.interpolation})')

        return repr_str


@TRANSFORMS.register_module()
class NumpyPad(BaseTransform):
    """Numpy Padding.

    In this augmentation, numpy padding is adopted to customize padding
    augmentation. Please carefully read the numpy manual in:
    https://numpy.org/doc/stable/reference/generated/numpy.pad.html

    If you just hope a single dimension to be padded, you must set ``padding``
    like this:

    ::

        padding = ((2, 2), (0, 0), (0, 0))

    In this case, if you adopt an input with three dimension, only the first
    dimension will be padded.

    Args:
        keys (Union[str, List[str]]): The images to be padded.
        padding (int | tuple(int)): Please refer to the args ``pad_width`` in
            ``numpy.pad``.
    """

    def __init__(self, keys, padding, **kwargs):
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys
        self.padding = padding
        self.kwargs = kwargs

    def transform(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        for key in self.keys:
            results[key] = np.pad(results[key], self.padding, **self.kwargs)

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += (
            f'(keys={self.keys}, padding={self.padding}, kwargs={self.kwargs})'
        )
        return repr_str
