# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import albumentations as albu
from mmcv.transforms import BaseTransform

from mmagic.registry import TRANSFORMS


@TRANSFORMS.register_module()
class PairedAlbuTransForms(BaseTransform):
    """PairedAlbuTransForms augmentation.

    Apply the same AlbuTransforms augmentation to paired images.
    """

    def __init__(self,
                 size: int,
                 lq_key: str = 'img',
                 gt_key: str = 'gt',
                 scope: str = 'geometric',
                 crop: str = 'random',
                 p: float = 0.5):
        self.size = size
        self.lq_key = lq_key
        self.gt_key = gt_key
        self.scope = scope
        self.crop = crop
        self.p = p
        augs = {
            'weak':
            albu.Compose([
                albu.HorizontalFlip(),
            ], p=self.p),
            'geometric':
            albu.OneOf([
                albu.HorizontalFlip(always_apply=True),
                albu.ShiftScaleRotate(always_apply=True),
                albu.Transpose(always_apply=True),
                albu.OpticalDistortion(always_apply=True),
                albu.ElasticTransform(always_apply=True),
            ],
                       p=self.p)
        }
        aug_fn = augs[self.scope]
        crop_fn = {
            'random': albu.RandomCrop(self.size, self.size, always_apply=True),
            'center': albu.CenterCrop(self.size, self.size, always_apply=True)
        }[self.crop]
        pad = albu.PadIfNeeded(self.size, self.size)

        self.pipeline = albu.Compose([aug_fn, pad, crop_fn],
                                     additional_targets={'target': 'image'})

    def transform(self, results):
        """processing input results according to `self.pipeline`.

        Args:
            results (dict): contains the processed data
            through the transform pipeline.

        Returns:
            results: the processed data.
        """
        r = self.pipeline(
            image=results[self.lq_key], target=results[self.gt_key])
        results[self.lq_key] = r['image']
        results[self.gt_key] = r['target']
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(size={self.size}, '
                     f'lq_key={self.lq_key}, '
                     f'gt_key={self.gt_key}, '
                     f'scope={self.scope}, '
                     f'crop={self.crop}, '
                     f'p={self.p})')

        return repr_str


@TRANSFORMS.register_module()
class AlbuTransForms(BaseTransform):
    """AlbuTransForms augmentation.

    Apply the same AlbuTransForms augmentation to the input images.
    """

    def __init__(self,
                 size: int,
                 keys: List,
                 scope: str = 'geometric',
                 crop: str = 'random',
                 p: float = 0.5):
        self.size = size
        self.keys = keys
        self.scope = scope
        self.crop = crop
        self.p = p
        augs = {
            'weak':
            albu.Compose([
                albu.HorizontalFlip(),
            ]),
            'geometric':
            albu.OneOf([
                albu.HorizontalFlip(always_apply=True),
                albu.ShiftScaleRotate(always_apply=True),
                albu.Transpose(always_apply=True),
                albu.OpticalDistortion(always_apply=True),
                albu.ElasticTransform(always_apply=True),
            ],
                       p=self.p)
        }
        aug_fn = augs[self.scope]
        crop_fn = {
            'random': albu.RandomCrop(self.size, self.size, always_apply=True),
            'center': albu.CenterCrop(self.size, self.size, always_apply=True)
        }[self.crop]
        pad = albu.PadIfNeeded(self.size, self.size)

        self.pipeline = albu.Compose([aug_fn, pad, crop_fn])

    def transform(self, results):
        """processing input results according to `self.pipeline`.

        Args:
            results (dict): contains the processed data
            through the transform pipeline.

        Returns:
            results: the processed data.
        """
        for key in self.keys:
            r = self.pipeline(image=results[key])
            results[key] = r['image']
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(size={self.size}, '
                     f'keys={self.keys}, '
                     f'scope={self.scope}, '
                     f'crop={self.crop}, '
                     f'p={self.p})')

        return repr_str


@TRANSFORMS.register_module()
class PairedAlbuNormalize(BaseTransform):
    """PairedAlbuNormalize augmentation.

    Apply the same AlbuNormalize augmentation to the paired images.
    """

    def __init__(self,
                 lq_key: str,
                 gt_key: str,
                 mean=(0.5, 0.5, 0.5),
                 std=(0.5, 0.5, 0.5),
                 max_pixel_value: float = 255.0,
                 always_apply: bool = False,
                 p: float = 1.0):
        self.lq_key = lq_key
        self.gt_key = gt_key
        self.mean = mean
        self.std = std
        self.max_pixel_value = max_pixel_value
        self.always_apply = always_apply
        self.p = p
        normalize = albu.Normalize(
            mean=self.mean,
            std=self.std,
            max_pixel_value=self.max_pixel_value,
            always_apply=self.always_apply,
            p=self.p)
        self.normalize = albu.Compose([normalize],
                                      additional_targets={'target': 'image'})

    def transform(self, results):
        """processing input results according to `self.normalize`.

        Args:
            results (dict): contains the processed data
            through the transform pipeline.

        Returns:
            results: the processed data.
        """
        if self.gt_key not in results.keys():
            r = self.normalize(image=results[self.lq_key])
        else:
            r = self.normalize(
                image=results[self.lq_key], target=results[self.gt_key])
            results[self.gt_key] = r['target']
        results[self.lq_key] = r['image']

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(lq_key={self.lq_key}, '
                     f'gt_key={self.gt_key}, '
                     f'mean={self.mean}, '
                     f'std={self.std}, '
                     f'max_pixel_value={self.max_pixel_value}, '
                     f'always_apply={self.always_apply}, '
                     f'p={self.p}) ')

        return repr_str


@TRANSFORMS.register_module()
class AlbuNormalize(BaseTransform):
    """AlbuNormalize augmentation.

    Apply the same AlbuNormalize augmentation to the input images.
    """

    def __init__(self,
                 keys: List,
                 mean=(0.5, 0.5, 0.5),
                 std=(0.5, 0.5, 0.5),
                 max_pixel_value: float = 255.0,
                 always_apply: bool = False,
                 p: float = 1.0):
        self.keys = keys
        self.mean = mean
        self.std = std
        self.max_pixel_value = max_pixel_value
        self.always_apply = always_apply
        self.p = p
        normalize = albu.Normalize(
            mean=self.mean,
            std=self.std,
            max_pixel_value=self.max_pixel_value,
            always_apply=self.always_apply,
            p=self.p)
        self.normalize = albu.Compose([normalize])

    def transform(self, results):
        """processing input results according to `self.normalize`.

        Args:
            results (dict): contains the processed data
            through the transform pipeline.

        Returns:
            results: the processed data.
        """
        for key in self.keys:
            r = self.normalize(image=results[key])
            results[key] = r['image']
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(keys={self.keys}, '
                     f'mean={self.mean}, '
                     f'std={self.std}, '
                     f'max_pixel_value={self.max_pixel_value}, '
                     f'always_apply={self.always_apply}, '
                     f'p={self.p}) ')

        return repr_str


def _resolve_aug_fn(name):
    d = {
        'cutout': albu.Cutout,
        'rgb_shift': albu.RGBShift,
        'hsv_shift': albu.HueSaturationValue,
        'motion_blur': albu.MotionBlur,
        'median_blur': albu.MedianBlur,
        'snow': albu.RandomSnow,
        'shadow': albu.RandomShadow,
        'fog': albu.RandomFog,
        'brightness_contrast': albu.RandomBrightnessContrast,
        'gamma': albu.RandomGamma,
        'sun_flare': albu.RandomSunFlare,
        'sharpen': albu.Sharpen,
        'jpeg': albu.ImageCompression,
        'gray': albu.ToGray,
        'pixelize': albu.Downscale,
        # ToDo: partial gray
    }
    return d[name]


@TRANSFORMS.register_module()
class AlbuCorruptFunction(BaseTransform):
    """AlbuCorruptFunction augmentation.

    Apply the same AlbuCorruptFunction augmentation to the input images.
    """

    def __init__(self, keys: List[str], config: List[dict], p: float = 1.0):
        self.keys = keys
        self.config = config
        self.p = p
        augs = []
        for aug_params in self.config:
            name = aug_params.pop('name')
            cls = _resolve_aug_fn(name)
            prob = aug_params.pop('prob') if 'prob' in aug_params else .5
            augs.append(cls(p=prob, **aug_params))

        self.augs = albu.OneOf(augs, p=self.p)

    def transform(self, results):
        """processing input results according to `self.augs`.

        Args:
            results (dict): contains the processed data
            through the transform pipeline.

        Returns:
            results: the processed data.
        """
        for key in self.keys:
            results[key] = self.augs(image=results[key])['image']
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(keys={self.keys}, '
                     f'config={self.config}, '
                     f'p={self.p}) ')

        return repr_str
