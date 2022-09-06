# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
import numbers
import os
import os.path as osp
import random

import cv2
import mmcv
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from ..registry import PIPELINES


@PIPELINES.register_module()
class Resize:
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

    All keys in "keys" should have the same shape. "test_trans" is used to
    record the test transformation to align the input's shape.

    Args:
        keys (list[str]): The images to be resized.
        scale (float | tuple[int]): If scale is tuple[int], target spatial
            size (h, w). Otherwise, target spatial size is scaled by input
            size.
            Note that when it is used, `size_factor` and `max_size` are
            useless. Default: None
        keep_ratio (bool): If set to True, images will be resized without
            changing the aspect ratio. Otherwise, it will resize images to a
            given size. Default: False.
            Note that it is used togher with `scale`.
        size_factor (int): Let the output shape be a multiple of size_factor.
            Default:None.
            Note that when it is used, `scale` should be set to None and
            `keep_ratio` should be set to False.
        max_size (int): The maximum size of the longest side of the output.
            Default:None.
            Note that it is used togher with `size_factor`.
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
                 keys,
                 scale=None,
                 keep_ratio=False,
                 size_factor=None,
                 max_size=None,
                 interpolation='bilinear',
                 backend=None,
                 output_keys=None):
        assert keys, 'Keys should not be empty.'
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
        elif mmcv.is_tuple_of(scale, int):
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
            self.scale_factor = np.array((w_scale, h_scale), dtype=np.float32)
        return img

    def __call__(self, results):
        """Call function.

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
            results[out_key] = self._resize(results[key])
            if len(results[out_key].shape) == 2:
                results[out_key] = np.expand_dims(results[out_key], axis=2)

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


@PIPELINES.register_module()
class RandomRotation:
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
        elif not mmcv.is_tuple_of(degrees, (int, float)):
            raise TypeError(f'Degrees must be float | int or tuple of float | '
                            'int, but got '
                            f'{type(degrees)}.')

        self.keys = keys
        self.degrees = degrees

    def __call__(self, results):
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


@PIPELINES.register_module()
class Flip:
    """Flip the input data with a probability.

    Reverse the order of elements in the given data with a specific direction.
    The shape of the data is preserved, but the elements are reordered.
    Required keys are the keys in attributes "keys", added or modified keys are
    "flip", "flip_direction" and the keys in attributes "keys".
    It also supports flipping a list of images with the same flip.

    Args:
        keys (list[str]): The images to be flipped.
        flip_ratio (float): The propability to flip the images.
        direction (str): Flip images horizontally or vertically. Options are
            "horizontal" | "vertical". Default: "horizontal".
    """
    _directions = ['horizontal', 'vertical']

    def __init__(self, keys, flip_ratio=0.5, direction='horizontal'):
        if direction not in self._directions:
            raise ValueError(f'Direction {direction} is not supported.'
                             f'Currently support ones are {self._directions}')
        self.keys = keys
        self.flip_ratio = flip_ratio
        self.direction = direction

    def __call__(self, results):
        """Call function.

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

        results['flip'] = flip
        results['flip_direction'] = self.direction

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(keys={self.keys}, flip_ratio={self.flip_ratio}, '
                     f'direction={self.direction})')
        return repr_str


@PIPELINES.register_module()
class Pad:
    """Pad the images to align with network downsample factor for testing.

    See `Reshape` for more explanation. `numpy.pad` is used for the pad
    operation.
    Required keys are the keys in attribute "keys", added or
    modified keys are "test_trans" and the keys in attribute
    "keys". All keys in "keys" should have the same shape. "test_trans" is used
    to record the test transformation to align the input's shape.

    Args:
        keys (list[str]): The images to be padded.
        ds_factor (int): Downsample factor of the network. The height and
            weight will be padded to a multiple of ds_factor. Default: 32.
        kwargs (option): any keyword arguments to be passed to `numpy.pad`.
    """

    def __init__(self, keys, ds_factor=32, **kwargs):
        self.keys = keys
        self.ds_factor = ds_factor
        self.kwargs = kwargs

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        h, w = results[self.keys[0]].shape[:2]

        new_h = self.ds_factor * ((h - 1) // self.ds_factor + 1)
        new_w = self.ds_factor * ((w - 1) // self.ds_factor + 1)

        pad_h = new_h - h
        pad_w = new_w - w
        if new_h != h or new_w != w:
            pad_width = ((0, pad_h), (0, pad_w), (0, 0))
            for key in self.keys:
                results[key] = np.pad(results[key],
                                      pad_width[:results[key].ndim],
                                      **self.kwargs)
        results['pad'] = (pad_h, pad_w)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        kwargs_str = ', '.join(
            [f'{key}={val}' for key, val in self.kwargs.items()])
        repr_str += (f'(keys={self.keys}, ds_factor={self.ds_factor}, '
                     f'{kwargs_str})')
        return repr_str


@PIPELINES.register_module()
class RandomAffine:
    """Apply random affine to input images.

    This class is adopted from
    https://github.com/pytorch/vision/blob/v0.5.0/torchvision/transforms/
    transforms.py#L1015
    It should be noted that in
    https://github.com/Yaoyi-Li/GCA-Matting/blob/master/dataloader/
    data_generator.py#L70
    random flip is added. See explanation of `flip_ratio` below.
    Required keys are the keys in attribute "keys", modified keys
    are keys in attribute "keys".

    Args:
        keys (Sequence[str]): The images to be affined.
        degrees (float | tuple[float]): Range of degrees to select from. If it
            is a float instead of a tuple like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to deactivate rotations.
        translate (tuple, optional): Tuple of maximum absolute fraction for
            horizontal and vertical translations. For example translate=(a, b),
            then horizontal shift is randomly sampled in the range
            -img_width * a < dx < img_width * a and vertical shift is randomly
            sampled in the range -img_height * b < dy < img_height * b.
            Default: None.
        scale (tuple, optional): Scaling factor interval, e.g (a, b), then
            scale is randomly sampled from the range a <= scale <= b.
            Default: None.
        shear (float | tuple[float], optional): Range of shear degrees to
            select from. If shear is a float, a shear parallel to the x axis
            and a shear parallel to the y axis in the range (-shear, +shear)
            will be applied. Else if shear is a tuple of 2 values, a x-axis
            shear and a y-axis shear in (shear[0], shear[1]) will be applied.
            Default: None.
        flip_ratio (float, optional): Probability of the image being flipped.
            The flips in horizontal direction and vertical direction are
            independent. The image may be flipped in both directions.
            Default: None.
    """

    def __init__(self,
                 keys,
                 degrees,
                 translate=None,
                 scale=None,
                 shear=None,
                 flip_ratio=None):
        self.keys = keys
        if isinstance(degrees, numbers.Number):
            assert degrees >= 0, ('If degrees is a single number, '
                                  'it must be positive.')
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, tuple) and len(degrees) == 2, \
                'degrees should be a tuple and it must be of length 2.'
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, tuple) and len(translate) == 2, \
                'translate should be a tuple and it must be of length 2.'
            for t in translate:
                assert 0.0 <= t <= 1.0, ('translation values should be '
                                         'between 0 and 1.')
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, tuple) and len(scale) == 2, \
                'scale should be a tuple and it must be of length 2.'
            for s in scale:
                assert s > 0, 'scale values should be positive.'
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                assert shear >= 0, ('If shear is a single number, '
                                    'it must be positive.')
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, tuple) and len(shear) == 2, \
                    'shear should be a tuple and it must be of length 2.'
                # X-Axis and Y-Axis shear with (min, max)
                self.shear = shear
        else:
            self.shear = shear

        if flip_ratio is not None:
            assert isinstance(flip_ratio,
                              float), 'flip_ratio should be a float.'
            self.flip_ratio = flip_ratio
        else:
            self.flip_ratio = 0

    @staticmethod
    def _get_params(degrees, translate, scale_ranges, shears, flip_ratio,
                    img_size):
        """Get parameters for affine transformation.

        Returns:
            paras (tuple): Params to be passed to the affine transformation.
        """
        angle = np.random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(np.random.uniform(-max_dx, max_dx)),
                            np.round(np.random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = (np.random.uniform(scale_ranges[0], scale_ranges[1]),
                     np.random.uniform(scale_ranges[0], scale_ranges[1]))
        else:
            scale = (1.0, 1.0)

        if shears is not None:
            shear = np.random.uniform(shears[0], shears[1])
        else:
            shear = 0.0

        # Because `flip` is used as a multiplier in line 479 and 480,
        # so -1 stands for flip and 1 stands for no flip. Thus `flip`
        # should be an 'inverse' flag as the result of the comparison.
        # See https://github.com/open-mmlab/mmediting/pull/799 for more detail
        flip = (np.random.rand(2) > flip_ratio).astype(np.int32) * 2 - 1

        return angle, translations, scale, shear, flip

    @staticmethod
    def _get_inverse_affine_matrix(center, angle, translate, scale, shear,
                                   flip):
        """Helper method to compute inverse matrix for affine transformation.

        As it is explained in PIL.Image.rotate, we need compute INVERSE of
        affine transformation matrix: M = T * C * RSS * C^-1 where
        T is translation matrix:
            [1, 0, tx | 0, 1, ty | 0, 0, 1];
        C is translation matrix to keep center:
            [1, 0, cx | 0, 1, cy | 0, 0, 1];
        RSS is rotation with scale and shear matrix.

        It is different from the original function in torchvision.
        1. The order are changed to flip -> scale -> rotation -> shear.
        2. x and y have different scale factors.
        RSS(shear, a, scale, f) =
            [ cos(a + shear)*scale_x*f -sin(a + shear)*scale_y     0]
            [ sin(a)*scale_x*f          cos(a)*scale_y             0]
            [     0                       0                        1]
        Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1.
        """

        angle = math.radians(angle)
        shear = math.radians(shear)
        scale_x = 1.0 / scale[0] * flip[0]
        scale_y = 1.0 / scale[1] * flip[1]

        # Inverted rotation matrix with scale and shear
        d = math.cos(angle + shear) * math.cos(angle) + math.sin(
            angle + shear) * math.sin(angle)
        matrix = [
            math.cos(angle) * scale_x,
            math.sin(angle + shear) * scale_x, 0, -math.sin(angle) * scale_y,
            math.cos(angle + shear) * scale_y, 0
        ]
        matrix = [m / d for m in matrix]

        # Apply inverse of translation and of center translation:
        # RSS^-1 * C^-1 * T^-1
        matrix[2] += matrix[0] * (-center[0] - translate[0]) + matrix[1] * (
            -center[1] - translate[1])
        matrix[5] += matrix[3] * (-center[0] - translate[0]) + matrix[4] * (
            -center[1] - translate[1])

        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        matrix[2] += center[0]
        matrix[5] += center[1]

        return matrix

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        h, w = results[self.keys[0]].shape[:2]
        # if image is too small, set degree to 0 to reduce introduced dark area
        if np.maximum(h, w) < 1024:
            params = self._get_params((0, 0), self.translate, self.scale,
                                      self.shear, self.flip_ratio, (h, w))
        else:
            params = self._get_params(self.degrees, self.translate, self.scale,
                                      self.shear, self.flip_ratio, (h, w))

        center = (w * 0.5 - 0.5, h * 0.5 - 0.5)
        M = self._get_inverse_affine_matrix(center, *params)
        M = np.array(M).reshape((2, 3))

        for key in self.keys:
            results[key] = cv2.warpAffine(
                results[key],
                M, (w, h),
                flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(keys={self.keys}, degrees={self.degrees}, '
                     f'translate={self.translate}, scale={self.scale}, '
                     f'shear={self.shear}, flip_ratio={self.flip_ratio})')
        return repr_str


@PIPELINES.register_module()
class RandomJitter:
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

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        fg, alpha = results['fg'], results['alpha']

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


@PIPELINES.register_module()
class ColorJitter:
    """An interface for torch color jitter so that it can be invoked in
    mmediting pipeline.

    Randomly change the brightness, contrast and saturation of an image.
    Modified keys are the attributes specified in "keys".

    Args:
        keys (list[str]): The images to be resized.
        channel_order (str): Order of channel, candidates are 'bgr' and 'rgb'.
            Default: 'rgb'.

    Notes: ``**kwards`` follows the args list of
        ``torchvision.transforms.ColorJitter``.

        brightness (float or tuple of float (min, max)): How much to jitter
            brightness. brightness_factor is chosen uniformly from
            [max(0, 1 - brightness), 1 + brightness] or the given [min, max].
            Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter
            contrast. contrast_factor is chosen uniformly from
            [max(0, 1 - contrast), 1 + contrast] or the given [min, max].
            Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter
            saturation. saturation_factor is chosen uniformly from
            [max(0, 1 - saturation), 1 + saturation] or the given [min, max].
            Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given
            [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(self, keys, channel_order='rgb', **kwargs):
        assert keys, 'Keys should not be empty.'
        assert 'to_rgb' not in kwargs, (
            '`to_rgb` is not support in ColorJitter, '
            "which is replaced by `channel_order` ('rgb' or 'bgr')")

        self.keys = keys
        self.channel_order = channel_order
        self.transform = transforms.ColorJitter(**kwargs)

    def _color_jitter(self, image, this_seed):

        if self.channel_order.lower() == 'bgr':
            image = image[..., ::-1]

        image = Image.fromarray(image)
        torch.manual_seed(this_seed)
        image = self.transform(image)
        image = np.asarray(image)

        if self.channel_order.lower() == 'bgr':
            image = image[..., ::-1]

        return image

    def __call__(self, results):

        this_seed = random.randint(0, 2**32)

        for k in self.keys:
            if isinstance(results[k], list):
                results[k] = [
                    self._color_jitter(v, this_seed) for v in results[k]
                ]
            else:
                results[k] = self._color_jitter(results[k], this_seed)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(keys={self.keys}, channel_order={self.channel_order}, '
                     f'brightness={self.transform.brightness}, '
                     f'contrast={self.transform.contrast}, '
                     f'saturation={self.transform.saturation}, '
                     f'hue={self.transform.hue})')

        return repr_str


class BinarizeImage:
    """Binarize image.

    Args:
        keys (Sequence[str]): The images to be binarized.
        binary_thr (float): Threshold for binarization.
        to_int (bool): If True, return image as int32, otherwise
            return image as float32.
    """

    def __init__(self, keys, binary_thr, to_int=False):
        self.keys = keys
        self.binary_thr = binary_thr
        self.to_int = to_int

    def _binarize(self, img):
        type_ = np.float32 if not self.to_int else np.int32
        img = (img[..., :] > self.binary_thr).astype(type_)

        return img

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        for k in self.keys:
            results[k] = self._binarize(results[k])

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(keys={self.keys}, binary_thr={self.binary_thr}, '
                     f'to_int={self.to_int})')

        return repr_str


@PIPELINES.register_module()
class RandomMaskDilation:
    """Randomly dilate binary masks.

    Args:
        keys (Sequence[str]): The images to be resized.
        get_binary (bool): If True, according to binary_thr, reset final
            output as binary mask. Otherwise, return masks directly.
        binary_thr (float): Threshold for obtaining binary mask.
        kernel_min (int): Min size of dilation kernel.
        kernel_max (int): Max size of dilation kernel.
    """

    def __init__(self, keys, binary_thr=0., kernel_min=9, kernel_max=49):
        self.keys = keys
        self.kernel_min = kernel_min
        self.kernel_max = kernel_max
        self.binary_thr = binary_thr

    def _random_dilate(self, img):
        kernel_size = np.random.randint(self.kernel_min, self.kernel_max + 1)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        dilate_kernel_size = kernel_size
        img_ = cv2.dilate(img, kernel, iterations=1)

        img_ = (img_ > self.binary_thr).astype(np.float32)

        return img_, dilate_kernel_size

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        for k in self.keys:
            results[k], d_kernel = self._random_dilate(results[k])
            if len(results[k].shape) == 2:
                results[k] = np.expand_dims(results[k], axis=2)
            results[k + '_dilate_kernel_size'] = d_kernel

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(keys={self.keys}, kernel_min={self.kernel_min}, '
                     f'kernel_max={self.kernel_max})')

        return repr_str


@PIPELINES.register_module()
class RandomTransposeHW:
    """Randomly transpose images in H and W dimensions with a probability.

    (TransposeHW = horizontal flip + anti-clockwise rotatation by 90 degrees)
    When used with horizontal/vertical flips, it serves as a way of rotation
    augmentation.
    It also supports randomly transposing a list of images.

    Required keys are the keys in attributes "keys", added or modified keys are
    "transpose" and the keys in attributes "keys".

    Args:
        keys (list[str]): The images to be transposed.
        transpose_ratio (float): The propability to transpose the images.
    """

    def __init__(self, keys, transpose_ratio=0.5):
        self.keys = keys
        self.transpose_ratio = transpose_ratio

    def __call__(self, results):
        """Call function.

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


@PIPELINES.register_module()
class GenerateFrameIndiceswithPadding:
    """Generate frame index with padding for REDS dataset and Vid4 dataset
    during testing.

    Required keys: lq_path, gt_path, key, num_input_frames, max_frame_num
    Added or modified keys: lq_path, gt_path

    Args:
         padding (str): padding mode, one of
            'replicate' | 'reflection' | 'reflection_circle' | 'circle'.

            Examples: current_idx = 0, num_input_frames = 5
            The generated frame indices under different padding mode:

                replicate: [0, 0, 0, 1, 2]
                reflection: [2, 1, 0, 1, 2]
                reflection_circle: [4, 3, 0, 1, 2]
                circle: [3, 4, 0, 1, 2]

        filename_tmpl (str): Template for file name. Default: '{:08d}'.
    """

    def __init__(self, padding, filename_tmpl='{:08d}'):
        if padding not in ('replicate', 'reflection', 'reflection_circle',
                           'circle'):
            raise ValueError(f'Wrong padding mode {padding}.'
                             'Should be "replicate", "reflection", '
                             '"reflection_circle",  "circle"')
        self.padding = padding
        self.filename_tmpl = filename_tmpl

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        clip_name, frame_name = results['key'].split(os.sep)
        current_idx = int(frame_name)
        max_frame_num = results['max_frame_num'] - 1  # start from 0
        num_input_frames = results['num_input_frames']
        num_pad = num_input_frames // 2

        frame_list = []
        for i in range(current_idx - num_pad, current_idx + num_pad + 1):
            if i < 0:
                if self.padding == 'replicate':
                    pad_idx = 0
                elif self.padding == 'reflection':
                    pad_idx = -i
                elif self.padding == 'reflection_circle':
                    pad_idx = current_idx + num_pad - i
                else:
                    pad_idx = num_input_frames + i
            elif i > max_frame_num:
                if self.padding == 'replicate':
                    pad_idx = max_frame_num
                elif self.padding == 'reflection':
                    pad_idx = max_frame_num * 2 - i
                elif self.padding == 'reflection_circle':
                    pad_idx = (current_idx - num_pad) - (i - max_frame_num)
                else:
                    pad_idx = i - num_input_frames
            else:
                pad_idx = i
            frame_list.append(pad_idx)

        lq_path_root = results['lq_path']
        gt_path_root = results['gt_path']
        lq_paths = [
            osp.join(lq_path_root, clip_name,
                     f'{self.filename_tmpl.format(idx)}.png')
            for idx in frame_list
        ]
        gt_paths = [osp.join(gt_path_root, clip_name, f'{frame_name}.png')]
        results['lq_path'] = lq_paths
        results['gt_path'] = gt_paths

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f"(padding='{self.padding}')"
        return repr_str


@PIPELINES.register_module()
class GenerateFrameIndices:
    """Generate frame index for REDS datasets. It also performs temporal
    augmention with random interval.

    Required keys: lq_path, gt_path, key, num_input_frames
    Added or modified keys:  lq_path, gt_path, interval, reverse

    Args:
        interval_list (list[int]): Interval list for temporal augmentation.
            It will randomly pick an interval from interval_list and sample
            frame index with the interval.
        frames_per_clip(int): Number of frames per clips. Default: 99 for
            REDS dataset.
    """

    def __init__(self, interval_list, frames_per_clip=99):
        self.interval_list = interval_list
        self.frames_per_clip = frames_per_clip

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        clip_name, frame_name = results['key'].split(
            os.sep)  # key example: 000/00000000
        center_frame_idx = int(frame_name)
        num_half_frames = results['num_input_frames'] // 2

        max_frame_num = results.get('max_frame_num', self.frames_per_clip + 1)
        frames_per_clip = min(self.frames_per_clip, max_frame_num - 1)

        interval = np.random.choice(self.interval_list)
        # ensure not exceeding the borders
        start_frame_idx = center_frame_idx - num_half_frames * interval
        end_frame_idx = center_frame_idx + num_half_frames * interval
        while (start_frame_idx < 0) or (end_frame_idx > frames_per_clip):
            center_frame_idx = np.random.randint(0, frames_per_clip + 1)
            start_frame_idx = center_frame_idx - num_half_frames * interval
            end_frame_idx = center_frame_idx + num_half_frames * interval
        frame_name = f'{center_frame_idx:08d}'
        neighbor_list = list(
            range(center_frame_idx - num_half_frames * interval,
                  center_frame_idx + num_half_frames * interval + 1, interval))

        lq_path_root = results['lq_path']
        gt_path_root = results['gt_path']
        lq_path = [
            osp.join(lq_path_root, clip_name, f'{v:08d}.png')
            for v in neighbor_list
        ]
        gt_path = [osp.join(gt_path_root, clip_name, f'{frame_name}.png')]
        results['lq_path'] = lq_path
        results['gt_path'] = gt_path
        results['interval'] = interval

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(interval_list={self.interval_list}, '
                     f'frames_per_clip={self.frames_per_clip})')
        return repr_str


@PIPELINES.register_module()
class TemporalReverse:
    """Reverse frame lists for temporal augmentation.

    Required keys are the keys in attributes "lq" and "gt",
    added or modified keys are "lq", "gt" and "reverse".

    Args:
        keys (list[str]): The frame lists to be reversed.
        reverse_ratio (float): The propability to reverse the frame lists.
            Default: 0.5.
    """

    def __init__(self, keys, reverse_ratio=0.5):
        self.keys = keys
        self.reverse_ratio = reverse_ratio

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        reverse = np.random.random() < self.reverse_ratio

        if reverse:
            for key in self.keys:
                results[key].reverse()

        results['reverse'] = reverse

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(keys={self.keys}, reverse_ratio={self.reverse_ratio})'
        return repr_str


@PIPELINES.register_module()
class GenerateSegmentIndices:
    """Generate frame indices for a segment. It also performs temporal
    augmention with random interval.

    Required keys: lq_path, gt_path, key, num_input_frames, sequence_length
    Added or modified keys:  lq_path, gt_path, interval, reverse

    Args:
        interval_list (list[int]): Interval list for temporal augmentation.
            It will randomly pick an interval from interval_list and sample
            frame index with the interval.
        start_idx (int): The index corresponds to the first frame in the
            sequence. Default: 0.
        filename_tmpl (str): Template for file name. Default: '{:08d}.png'.
    """

    def __init__(self, interval_list, start_idx=0, filename_tmpl='{:08d}.png'):
        self.interval_list = interval_list
        self.filename_tmpl = filename_tmpl
        self.start_idx = start_idx

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        # key example: '000', 'calendar' (sequence name)
        clip_name = results['key']
        interval = np.random.choice(self.interval_list)

        self.sequence_length = results['sequence_length']
        num_input_frames = results.get('num_input_frames',
                                       self.sequence_length)

        # randomly select a frame as start
        if self.sequence_length - num_input_frames * interval < 0:
            raise ValueError('The input sequence is not long enough to '
                             'support the current choice of [interval] or '
                             '[num_input_frames].')
        start_frame_idx = np.random.randint(
            0, self.sequence_length - num_input_frames * interval + 1)
        end_frame_idx = start_frame_idx + num_input_frames * interval
        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))
        neighbor_list = [v + self.start_idx for v in neighbor_list]

        # add the corresponding file paths
        lq_path_root = results['lq_path']
        gt_path_root = results['gt_path']
        lq_path = [
            osp.join(lq_path_root, clip_name, self.filename_tmpl.format(v))
            for v in neighbor_list
        ]
        gt_path = [
            osp.join(gt_path_root, clip_name, self.filename_tmpl.format(v))
            for v in neighbor_list
        ]

        results['lq_path'] = lq_path
        results['gt_path'] = gt_path
        results['interval'] = interval

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(interval_list={self.interval_list})')
        return repr_str


@PIPELINES.register_module()
class MirrorSequence:
    """Extend short sequences (e.g. Vimeo-90K) by mirroring the sequences.

    Given a sequence with N frames (x1, ..., xN), extend the sequence to
    (x1, ..., xN, xN, ..., x1).

    Args:
        keys (list[str]): The frame lists to be extended.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        for key in self.keys:
            if isinstance(results[key], list):
                results[key] = results[key] + results[key][::-1]
            else:
                raise TypeError('The input must be of class list[nparray]. '
                                f'Got {type(results[key])}.')

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(keys={self.keys})')
        return repr_str


@PIPELINES.register_module()
class CopyValues:
    """Copy the value of a source key to a destination key.

    It does the following: results[dst_key] = results[src_key] for
    (src_key, dst_key) in zip(src_keys, dst_keys).

    Added keys are the keys in the attribute "dst_keys".

    Args:
        src_keys (list[str]): The source keys.
        dst_keys (list[str]): The destination keys.
    """

    def __init__(self, src_keys, dst_keys):

        if not isinstance(src_keys, list) or not isinstance(dst_keys, list):
            raise AssertionError('"src_keys" and "dst_keys" must be lists.')

        if len(src_keys) != len(dst_keys):
            raise ValueError('"src_keys" and "dst_keys" should have the same'
                             'number of elements.')

        self.src_keys = src_keys
        self.dst_keys = dst_keys

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict with a key added/modified.
        """
        for (src_key, dst_key) in zip(self.src_keys, self.dst_keys):
            results[dst_key] = copy.deepcopy(results[src_key])

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(src_keys={self.src_keys})')
        repr_str += (f'(dst_keys={self.dst_keys})')
        return repr_str


@PIPELINES.register_module()
class Quantize:
    """Quantize and clip the image to [0, 1].

    It is assumed that the the input has range [0, 1].

    Modified keys are the attributes specified in "keys".

    Args:
        keys (list[str]): The keys whose values are clipped.
    """

    def __init__(self, keys):
        self.keys = keys

    def _quantize_clip(self, input_):
        is_single_image = False
        if isinstance(input_, np.ndarray):
            is_single_image = True
            input_ = [input_]

        # quantize and clip
        input_ = [np.clip((v * 255.0).round(), 0, 255) / 255. for v in input_]

        if is_single_image:
            input_ = input_[0]

        return input_

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict with the values of the specified keys are rounded
                and clipped.
        """

        for key in self.keys:
            results[key] = self._quantize_clip(results[key])

        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class UnsharpMasking:
    """Apply unsharp masking to an image or a sequence of images.

    Args:
        kernel_size (int): The kernel_size of the Gaussian kernel.
        sigma (float): The standard deviation of the Gaussian.
        weight (float): The weight of the "details" in the final output.
        threshold (float): Pixel differences larger than this value are
            regarded as "details".
        keys (list[str]): The keys whose values are processed.

    Added keys are "xxx_unsharp", where "xxx" are the attributes specified
    in "keys".
    """

    def __init__(self, kernel_size, sigma, weight, threshold, keys):
        if kernel_size % 2 == 0:
            raise ValueError('kernel_size must be an odd number, but '
                             f'got {kernel_size}.')

        self.kernel_size = kernel_size
        self.sigma = sigma
        self.weight = weight
        self.threshold = threshold
        self.keys = keys

        kernel = cv2.getGaussianKernel(kernel_size, sigma)
        self.kernel = np.matmul(kernel, kernel.transpose())

    def _unsharp_masking(self, imgs):
        is_single_image = False
        if isinstance(imgs, np.ndarray):
            is_single_image = True
            imgs = [imgs]

        outputs = []
        for img in imgs:
            residue = img - cv2.filter2D(img, -1, self.kernel)
            mask = np.float32(np.abs(residue) * 255 > self.threshold)
            soft_mask = cv2.filter2D(mask, -1, self.kernel)
            sharpened = np.clip(img + self.weight * residue, 0, 1)

            outputs.append(soft_mask * sharpened + (1 - soft_mask) * img)

        if is_single_image:
            outputs = outputs[0]

        return outputs

    def __call__(self, results):
        for key in self.keys:
            results[f'{key}_unsharp'] = self._unsharp_masking(results[key])

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(keys={self.keys}, kernel_size={self.kernel_size}, '
                     f'sigma={self.sigma}, weight={self.weight}, '
                     f'threshold={self.threshold})')
        return repr_str
