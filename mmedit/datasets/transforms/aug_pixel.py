# Copyright (c) OpenMMLab. All rights reserved.
import math
import numbers
import random
from typing import Dict

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from mmcv.transforms import BaseTransform
from PIL import Image

from mmedit.registry import TRANSFORMS


class BinarizeImage(BaseTransform):
    """Binarize image.

    Args:
        keys (Sequence[str]): The images to be binarized.
        binary_thr (float): Threshold for binarization.
        amin (int): Lower limits of pixel value.
        amx (int): Upper limits of pixel value.
        dtype (np.dtype): Set the data type of the output. Default: np.uint8
    """

    def __init__(self, keys, binary_thr, a_min=0, a_max=1, dtype=np.uint8):

        self.keys = keys
        self.binary_thr = binary_thr
        self.a_min = a_min
        self.a_max = a_max
        self.dtype = dtype

    def _binarize(self, img):
        """Binarize image.

        Args:
            img (np.ndarray): Input image.

        Returns:
            img (np.ndarray): Output image.
        """

        # Binarize to 0/1
        img = (img[..., :] > self.binary_thr).astype(np.uint8)

        if self.a_min != 0 or self.a_max != 1 or self.dtype != np.uint8:
            img = img * (self.a_max - self.a_min) + self.a_min
            img = img.astype(self.dtype)

        return img

    def transform(self, results):
        """The transform function of BinarizeImage.

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
        repr_str += (
            f'(keys={self.keys}, binary_thr={self.binary_thr}, '
            f'a_min={self.a_min}, a_max={self.a_max}, dtype={self.dtype})')

        return repr_str


@TRANSFORMS.register_module()
class Clip(BaseTransform):
    """Clip the pixels.

    Modified keys are the attributes specified in "keys".

    Args:
        keys (list[str]): The keys whose values are clipped.
        amin (int): Lower limits of pixel value.
        amx (int): Upper limits of pixel value.
    """

    def __init__(self, keys, a_min=0, a_max=255):

        self.keys = keys
        self.a_min = a_min
        self.a_max = a_max

    def _clip(self, input_):

        is_single_image = False
        if isinstance(input_, np.ndarray):
            is_single_image = True
            input_ = [input_]

        # clip
        input_ = [np.clip(v, self.a_min, self.a_max) for v in input_]

        if is_single_image:
            input_ = input_[0]

        return input_

    def transform(self, results):
        """transform function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict with the values of the specified keys are rounded
                and clipped.
        """

        for key in self.keys:
            results[key] = self._clip(results[key])

        return results

    def __repr__(self):

        result = self.__class__.__name__
        result += f'(a_min={self.a_min}, a_max={self.a_max})'

        return result


@TRANSFORMS.register_module()
class ColorJitter(BaseTransform):
    """An interface for torch color jitter so that it can be invoked in
    mmediting pipeline.

    Randomly change the brightness, contrast and saturation of an image.
    Modified keys are the attributes specified in "keys".

    Required Keys:

    - [KEYS]

    Modified Keys:

    - [KEYS]

    Args:
        keys (list[str]): The images to be resized.
        channel_order (str): Order of channel, candidates are 'bgr' and 'rgb'.
            Default: 'rgb'.

    Notes:

        ``**kwards`` follows the args list of
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
        self._transform = transforms.ColorJitter(**kwargs)

    def _color_jitter(self, image, this_seed):
        """Color Jitter Function.

        Args:
            image (np.ndarray): Image.
            this_seed (int): Seed of torch.

        Returns:
            image (np.ndarray): The output image.
        """

        if self.channel_order.lower() == 'bgr':
            image = image[..., ::-1]

        image = Image.fromarray(image)
        torch.manual_seed(this_seed)
        image = self._transform(image)
        image = np.asarray(image)

        if self.channel_order.lower() == 'bgr':
            image = image[..., ::-1]

        return image

    def transform(self, results: Dict) -> Dict:
        """The transform function of ColorJitter.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """

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
                     f'brightness={self._transform.brightness}, '
                     f'contrast={self._transform.contrast}, '
                     f'saturation={self._transform.saturation}, '
                     f'hue={self._transform.hue})')

        return repr_str


@TRANSFORMS.register_module()
class RandomAffine(BaseTransform):
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

    def transform(self, results):
        """transform function.

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
            ori_ndim = results[key].ndim
            results[key] = cv2.warpAffine(
                results[key],
                M, (w, h),
                flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP)

            if ori_ndim == 3 and results[key].ndim == 2:
                results[key] = results[key][..., None]

        return results

    def __repr__(self):

        repr_str = self.__class__.__name__
        repr_str += (f'(keys={self.keys}, degrees={self.degrees}, '
                     f'translate={self.translate}, scale={self.scale}, '
                     f'shear={self.shear}, flip_ratio={self.flip_ratio})')

        return repr_str


@TRANSFORMS.register_module()
class RandomMaskDilation(BaseTransform):
    """Randomly dilate binary masks.

    Args:
        keys (Sequence[str]): The images to be resized.
        binary_thr (float): Threshold for obtaining binary mask. Default: 0.
        kernel_min (int): Min size of dilation kernel. Default: 9.
        kernel_max (int): Max size of dilation kernel. Default: 49.
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

    def transform(self, results):
        """transform function.

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


@TRANSFORMS.register_module()
class UnsharpMasking(BaseTransform):
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
        """Unsharp masking function."""

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

    def transform(self, results):
        """transform function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        for key in self.keys:
            results[f'{key}_unsharp'] = self._unsharp_masking(results[key])

        return results

    def __repr__(self):

        repr_str = self.__class__.__name__
        repr_str += (f'(keys={self.keys}, kernel_size={self.kernel_size}, '
                     f'sigma={self.sigma}, weight={self.weight}, '
                     f'threshold={self.threshold})')

        return repr_str
