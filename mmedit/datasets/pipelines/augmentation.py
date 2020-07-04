import math
import numbers
import os.path as osp
import random

import cv2
import mmcv
import numpy as np

from ..registry import PIPELINES


@PIPELINES.register_module()
class Resize(object):
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
        scale (float | Tuple[int]): If scale is Tuple(int), target spatial
            size (h, w). Otherwise, target spatial size is scaled by input
            size. If any of scale is -1, we will rescale short edge.
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
    """

    def __init__(self,
                 keys,
                 scale=None,
                 keep_ratio=False,
                 size_factor=None,
                 max_size=None,
                 interpolation='bilinear'):
        assert keys, 'Keys should not be empty.'
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
        self.scale = scale
        self.size_factor = size_factor
        self.max_size = max_size
        self.keep_ratio = keep_ratio
        self.interpolation = interpolation

    def _resize(self, img):
        if self.keep_ratio:
            img, self.scale_factor = mmcv.imrescale(
                img,
                self.scale,
                return_scale=True,
                interpolation=self.interpolation)
        else:
            img, w_scale, h_scale = mmcv.imresize(
                img,
                self.scale,
                return_scale=True,
                interpolation=self.interpolation)
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
        for key in self.keys:
            results[key] = self._resize(results[key])
            if len(results[key].shape) == 2:
                results[key] = np.expand_dims(results[key], axis=2)

        results['scale_factor'] = self.scale_factor
        results['keep_ratio'] = self.keep_ratio
        results['interpolation'] = self.interpolation

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            f'(keys={self.keys}, scale={self.scale}, '
            f'keep_ratio={self.keep_ratio}, size_factor={self.size_factor}, '
            f'max_size={self.max_size},interpolation={self.interpolation})')
        return repr_str


@PIPELINES.register_module()
class Flip(object):
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
class Pad(object):
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
class RandomAffine(object):
    """Apply random affine to input images.

    This class is adopted from
    https://github.com/pytorch/vision/blob/v0.5.0/torchvision/transforms/transforms.py#L1015  # noqa
    It should be noted that in
    https://github.com/Yaoyi-Li/GCA-Matting/blob/master/dataloader/data_generator.py#L70  # noqa
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

        flip = (np.random.rand(2) < flip_ratio).astype(np.int) * 2 - 1

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

        center = (w * 0.5 + 0.5, h * 0.5 + 0.5)
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
class RandomJitter(object):
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


class BinarizeImage(object):
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
class RandomMaskDilation(object):
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
        kernel_size = random.randint(self.kernel_min, self.kernel_max)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        dilate_kernel_size = kernel_size
        img_ = cv2.dilate(img, kernel, iterations=1)
        h, w = img_.shape[:2]

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
class RandomTransposeHW(object):
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
class GenerateFrameIndiceswithPadding(object):
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
        clip_name, frame_name = results['key'].split('/')
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
class GenerateFrameIndices(object):
    """Generate frame index for REDS datasets. It also performs
    temporal augmention with random interval.

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
            '/')  # key example: 000/00000000
        center_frame_idx = int(frame_name)
        num_half_frames = results['num_input_frames'] // 2

        interval = np.random.choice(self.interval_list)
        # ensure not exceeding the borders
        start_frame_idx = center_frame_idx - num_half_frames * interval
        end_frame_idx = center_frame_idx + num_half_frames * interval
        while (start_frame_idx < 0) or (end_frame_idx > self.frames_per_clip):
            center_frame_idx = np.random.randint(0, self.frames_per_clip + 1)
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
class TemporalReverse(object):
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
