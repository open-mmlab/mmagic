# Copyright (c) OpenMMLab. All rights reserved.
import math
import random

import mmcv
import numpy as np
from mmcv.transforms import BaseTransform

from ..registry import TRANSFORMS


@TRANSFORMS.register_module()
class Crop(BaseTransform):
    """Crop data to specific size for training.

    Args:
        keys (Sequence[str]): The images to be cropped.
        crop_size (Tuple[int]): Target spatial size (h, w).
        random_crop (bool): If set to True, it will random crop
            image. Otherwise, it will work as center crop.
        is_pad_zeros (bool, optional): Whether to pad the image with 0 if
            crop_size is greater than image size. Default: False.
    """

    def __init__(self, keys, crop_size, random_crop=True, is_pad_zeros=False):

        if not mmcv.is_tuple_of(crop_size, int):
            raise TypeError(
                'Elements of crop_size must be int and crop_size must be'
                f' tuple, but got {type(crop_size[0])} in {type(crop_size)}')

        self.keys = keys
        self.crop_size = crop_size
        self.random_crop = random_crop
        self.is_pad_zeros = is_pad_zeros

    def _crop(self, data):

        if not isinstance(data, list):
            data_list = [data]
        else:
            data_list = data

        crop_bbox_list = []
        data_list_ = []

        for item in data_list:
            data_h, data_w = item.shape[:2]
            crop_h, crop_w = self.crop_size

            if self.is_pad_zeros:

                crop_y_offset, crop_x_offset = 0, 0

                if crop_h > data_h:
                    crop_y_offset = (crop_h - data_h) // 2
                if crop_w > data_w:
                    crop_x_offset = (crop_w - data_w) // 2

                if crop_y_offset > 0 or crop_x_offset > 0:
                    pad_width = [(2 * crop_y_offset, 2 * crop_y_offset),
                                 (2 * crop_x_offset, 2 * crop_x_offset)]
                    if item.ndim == 3:
                        pad_width.append((0, 0))
                    item = np.pad(
                        item,
                        tuple(pad_width),
                        mode='constant',
                        constant_values=0)

                data_h, data_w = item.shape[:2]

            crop_h = min(data_h, crop_h)
            crop_w = min(data_w, crop_w)

            if self.random_crop:
                x_offset = np.random.randint(0, data_w - crop_w + 1)
                y_offset = np.random.randint(0, data_h - crop_h + 1)
            else:
                x_offset = max(0, (data_w - crop_w)) // 2
                y_offset = max(0, (data_h - crop_h)) // 2

            crop_bbox = [x_offset, y_offset, crop_w, crop_h]
            item_ = item[y_offset:y_offset + crop_h,
                         x_offset:x_offset + crop_w, ...]
            crop_bbox_list.append(crop_bbox)
            data_list_.append(item_)

        if not isinstance(data, list):
            return data_list_[0], crop_bbox_list[0]
        return data_list_, crop_bbox_list

    def transform(self, results):
        """Transform function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """

        for k in self.keys:
            data_, crop_bbox = self._crop(results[k])
            results[k] = data_
            results[k + '_crop_bbox'] = crop_bbox
        results['crop_size'] = self.crop_size
        return results

    def __repr__(self):

        repr_str = self.__class__.__name__
        repr_str += (f'keys={self.keys}, crop_size={self.crop_size}, '
                     f'random_crop={self.random_crop}')

        return repr_str


@TRANSFORMS.register_module()
class CropLike(BaseTransform):
    """Crop/pad the image in the target_key according to the size of image
        in the reference_key .

    Args:
        target_key (str): The key needs to be cropped.
        reference_key (str | None): The reference key, need its size.
            Default: None.
    """

    def __init__(self, target_key, reference_key=None):

        assert reference_key and target_key
        self.target_key = target_key
        self.reference_key = reference_key

    def transform(self, results):
        """Transform function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.
                Require self.target_key and self.reference_key.

        Returns:
            dict: A dict containing the processed data and information.
                Modify self.target_key.
        """

        size = results[self.reference_key].shape
        old_image = results[self.target_key]
        old_size = old_image.shape
        h, w = old_size[:2]
        new_size = size[:2] + old_size[2:]
        h_cover, w_cover = min(h, size[0]), min(w, size[1])

        format_image = np.zeros(new_size, dtype=old_image.dtype)
        format_image[:h_cover, :w_cover] = old_image[:h_cover, :w_cover]
        results[self.target_key] = format_image

        return results

    def __repr__(self):

        return (self.__class__.__name__ + f' target_key={self.target_key}, ' +
                f'reference_key={self.reference_key}')


@TRANSFORMS.register_module()
class FixedCrop(BaseTransform):
    """Crop paired data (at a specific position) to specific size for training.

    Args:
        keys (Sequence[str]): The images to be cropped.
        crop_size (Tuple[int]): Target spatial size (h, w).
        crop_pos (Tuple[int]): Specific position (x, y). If set to None,
            random initialize the position to crop paired data batch.
    """

    def __init__(self, keys, crop_size, crop_pos=None):

        if not mmcv.is_tuple_of(crop_size, int):
            raise TypeError(
                'Elements of crop_size must be int and crop_size must be'
                f' tuple, but got {type(crop_size[0])} in {type(crop_size)}')
        if not mmcv.is_tuple_of(crop_pos, int) and (crop_pos is not None):
            raise TypeError(
                'Elements of crop_pos must be int and crop_pos must be'
                f' tuple or None, but got {type(crop_pos[0])} in '
                f'{type(crop_pos)}')

        self.keys = keys
        self.crop_size = crop_size
        self.crop_pos = crop_pos

    def _crop(self, data, x_offset, y_offset, crop_w, crop_h):

        crop_bbox = [x_offset, y_offset, crop_w, crop_h]
        data_ = data[y_offset:y_offset + crop_h, x_offset:x_offset + crop_w,
                     ...]

        return data_, crop_bbox

    def transform(self, results):
        """Transform function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """

        if isinstance(results[self.keys[0]], list):
            data_h, data_w = results[self.keys[0]][0].shape[:2]
        else:
            data_h, data_w = results[self.keys[0]].shape[:2]
        crop_h, crop_w = self.crop_size
        crop_h = min(data_h, crop_h)
        crop_w = min(data_w, crop_w)

        if self.crop_pos is None:
            x_offset = np.random.randint(0, data_w - crop_w + 1)
            y_offset = np.random.randint(0, data_h - crop_h + 1)
        else:
            x_offset, y_offset = self.crop_pos
            crop_w = min(data_w - x_offset, crop_w)
            crop_h = min(data_h - y_offset, crop_h)

        for k in self.keys:
            images = results[k]
            is_list = isinstance(images, list)
            if not is_list:
                images = [images]
            cropped_images = []
            crop_bbox = None
            for image in images:
                # In fixed crop for paired images, sizes should be the same
                if (image.shape[0] != data_h or image.shape[1] != data_w):
                    raise ValueError(
                        'The sizes of paired images should be the same. '
                        f'Expected ({data_h}, {data_w}), '
                        f'but got ({image.shape[0]}, '
                        f'{image.shape[1]}).')
                data_, crop_bbox = self._crop(image, x_offset, y_offset,
                                              crop_w, crop_h)
                cropped_images.append(data_)
            results[k + '_crop_bbox'] = crop_bbox
            if not is_list:
                cropped_images = cropped_images[0]
            results[k] = cropped_images
        results['crop_size'] = self.crop_size
        results['crop_pos'] = self.crop_pos

        return results

    def __repr__(self):

        repr_str = self.__class__.__name__
        repr_str += (f'keys={self.keys}, crop_size={self.crop_size}, '
                     f'crop_pos={self.crop_pos}')

        return repr_str


@TRANSFORMS.register_module()
class ModCrop(BaseTransform):
    """Mod crop gt images, used during testing.

    Required keys are "scale" and "gt",
    added or modified keys are "gt".
    """

    def transform(self, results):
        """Transform function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """

        img = results['gt'].copy()
        scale = results['scale']
        if img.ndim in [2, 3]:
            h, w = img.shape[0], img.shape[1]
            h_remainder, w_remainder = h % scale, w % scale
            img = img[:h - h_remainder, :w - w_remainder, ...]
        else:
            raise ValueError(f'Wrong img ndim: {img.ndim}.')
        results['gt'] = img

        return results


@TRANSFORMS.register_module()
class PairedRandomCrop(BaseTransform):
    """Paried random crop.

    It crops a pair of lq and gt images with corresponding locations.
    It also supports accepting lq list and gt list.
    Required keys are "scale", "lq", and "gt",
    added or modified keys are "lq" and "gt".

    Args:
        gt_patch_size (int): cropped gt patch size.
    """

    def __init__(self, gt_patch_size):

        self.gt_patch_size = gt_patch_size

    def transform(self, results):
        """Transform function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """

        scale = results['scale']
        lq_patch_size = self.gt_patch_size // scale

        lq_is_list = isinstance(results['lq'], list)
        if not lq_is_list:
            results['lq'] = [results['lq']]
        gt_is_list = isinstance(results['gt'], list)
        if not gt_is_list:
            results['gt'] = [results['gt']]

        h_lq, w_lq, _ = results['lq'][0].shape
        h_gt, w_gt, _ = results['gt'][0].shape

        if h_gt != h_lq * scale or w_gt != w_lq * scale:
            raise ValueError(
                f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x '
                f'multiplication of LQ ({h_lq}, {w_lq}).')
        if h_lq < lq_patch_size or w_lq < lq_patch_size:
            raise ValueError(
                f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                f'({lq_patch_size}, {lq_patch_size}). Please check '
                f'{results["lq_path"][0]} and {results["gt_path"][0]}.')

        # randomly choose top and left coordinates for lq patch
        top = np.random.randint(h_lq - lq_patch_size + 1)
        left = np.random.randint(w_lq - lq_patch_size + 1)
        # crop lq patch
        results['lq'] = [
            v[top:top + lq_patch_size, left:left + lq_patch_size, ...]
            for v in results['lq']
        ]
        # crop corresponding gt patch
        top_gt, left_gt = int(top * scale), int(left * scale)
        results['gt'] = [
            v[top_gt:top_gt + self.gt_patch_size,
              left_gt:left_gt + self.gt_patch_size, ...] for v in results['gt']
        ]

        if not lq_is_list:
            results['lq'] = results['lq'][0]
        if not gt_is_list:
            results['gt'] = results['gt'][0]

        return results

    def __repr__(self):

        repr_str = self.__class__.__name__
        repr_str += f'(gt_patch_size={self.gt_patch_size})'

        return repr_str


@TRANSFORMS.register_module()
class RandomResizedCrop(BaseTransform):
    """Crop data to random size and aspect ratio.

    A crop of a random proportion of the original image
    and a random aspect ratio of the original aspect ratio is made.
    The cropped image is finally resized to a given size specified
    by 'crop_size'. Modified keys are the attributes specified in "keys".

    This code is partially adopted from
    torchvision.transforms.RandomResizedCrop:
    [https://pytorch.org/vision/stable/_modules/torchvision/transforms/\
        transforms.html#RandomResizedCrop].

    Args:
        keys (list[str]): The images to be resized and random-cropped.
        crop_size (int | tuple[int]): Target spatial size (h, w).
        scale (tuple[float], optional): Range of the proportion of the original
            image to be cropped. Default: (0.08, 1.0).
        ratio (tuple[float], optional): Range of aspect ratio of the crop.
            Default: (3. / 4., 4. / 3.).
        interpolation (str, optional): Algorithm used for interpolation.
            It can be only either one of the following:
            "nearest" | "bilinear" | "bicubic" | "area" | "lanczos".
            Default: "bilinear".
    """

    def __init__(self,
                 keys,
                 crop_size,
                 scale=(0.08, 1.0),
                 ratio=(3. / 4., 4. / 3.),
                 interpolation='bilinear'):

        assert keys, 'Keys should not be empty.'
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        elif not mmcv.is_tuple_of(crop_size, int):
            raise TypeError('"crop_size" must be an integer '
                            'or a tuple of integers, but got '
                            f'{type(crop_size)}')
        if not mmcv.is_tuple_of(scale, float):
            raise TypeError('"scale" must be a tuple of float, '
                            f'but got {type(scale)}')
        if not mmcv.is_tuple_of(ratio, float):
            raise TypeError('"ratio" must be a tuple of float, '
                            f'but got {type(ratio)}')

        self.keys = keys
        self.crop_size = crop_size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation

    def get_params(self, data):
        """Get parameters for a random sized crop.

        Args:
            data (np.ndarray): Image of type numpy array to be cropped.

        Returns:
            A tuple containing the coordinates of the top left corner
            and the chosen crop size.
        """

        data_h, data_w = data.shape[:2]
        area = data_h * data_w

        for _ in range(10):
            target_area = random.uniform(*self.scale) * area
            log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            crop_w = int(round(math.sqrt(target_area * aspect_ratio)))
            crop_h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < crop_w <= data_w and 0 < crop_h <= data_h:
                top = random.randint(0, data_h - crop_h)
                left = random.randint(0, data_w - crop_w)
                return top, left, crop_h, crop_w

        # Fall back to center crop
        in_ratio = float(data_w) / float(data_h)
        if (in_ratio < min(self.ratio)):
            crop_w = data_w
            crop_h = int(round(crop_w / min(self.ratio)))
        elif (in_ratio > max(self.ratio)):
            crop_h = data_h
            crop_w = int(round(crop_h * max(self.ratio)))
        else:  # whole image
            crop_w = data_w
            crop_h = data_h
        top = (data_h - crop_h) // 2
        left = (data_w - crop_w) // 2

        return top, left, crop_h, crop_w

    def transform(self, results):
        """Transform function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """

        for k in self.keys:
            top, left, crop_h, crop_w = self.get_params(results[k])
            crop_bbox = [top, left, crop_w, crop_h]
            results[k] = results[k][top:top + crop_h, left:left + crop_w, ...]
            results[k] = mmcv.imresize(
                results[k],
                self.crop_size,
                return_scale=False,
                interpolation=self.interpolation)
            results[k + '_crop_bbox'] = crop_bbox

        return results

    def __repr__(self):

        repr_str = self.__class__.__name__
        repr_str += (f'(keys={self.keys}, crop_size={self.crop_size}, '
                     f'scale={self.scale}, ratio={self.ratio}, '
                     f'interpolation={self.interpolation})')

        return repr_str
