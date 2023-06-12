# Copyright (c) OpenMMLab. All rights reserved.
import copy
import inspect
from typing import List

import numpy as np
from mmcv.transforms import BaseTransform

from mmagic.registry import TRANSFORMS

try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None


@TRANSFORMS.register_module()
class Albumentations(BaseTransform):
    """Albumentation augmentation.

    Adds custom transformations from Albumentations library.
    Please, visit `https://github.com/albumentations-team/albumentations`
    and `https://albumentations.ai/docs/getting_started/transforms_and_targets`
    to get more information.

    An example of ``transforms`` is as followed:

    .. code-block::

        albu_transforms = [
            dict(
                type='Resize',
                height=100,
                width=100,
            ),
            dict(
                type='RandomFog',
                p=0.5,
            ),
            dict(
                type='RandomRain',
                p=0.5
            ),
            dict(
                type='RandomSnow',
                p=0.5,
            ),
        ]
        pipeline = [
            dict(
                type='LoadImageFromFile',
                key='img',
                color_type='color',
                channel_order='rgb',
                imdecode_backend='cv2'),
            dict(
                type='Albumentations',
                keys=['img'],
                transforms=albu_transforms),
            dict(type='PackInputs')
        ]

    Args:
        keys (list[str]): A list specifying the keys whose values are modified.
        transforms (list[dict]): A list of albu transformations.
    """

    def __init__(self, keys: List[str], transforms: List[dict]) -> None:

        if Compose is None:
            raise RuntimeError('Please install albumentations')

        self.keys = keys

        # Args will be modified later, copying it will be safer
        transforms = copy.deepcopy(transforms)
        self.transforms = transforms
        self.aug = Compose([self.albu_builder(t) for t in self.transforms])

    def albu_builder(self, cfg: dict) -> albumentations:
        """Import a module from albumentations.

        It inherits some of :func:`build_from_cfg` logic.

        Args:
            cfg (dict): Config dict. It should at least contain the key "type".

        Returns:
            obj: The constructed object.
        """

        assert isinstance(cfg, dict) and 'type' in cfg
        args = cfg.copy()
        obj_type = args.pop('type')
        if isinstance(obj_type, str):
            if albumentations is None:
                raise RuntimeError('Please install albumentations')
            obj_cls = getattr(albumentations, obj_type)
        elif inspect.isclass(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(
                f'type must be a str or valid type, but got {type(obj_type)}')

        if 'transforms' in args:
            args['transforms'] = [
                self.albu_builder(transform)
                for transform in args['transforms']
            ]

        return obj_cls(**args)

    def _apply_albu(self, imgs):
        is_single_image = False
        if isinstance(imgs, np.ndarray):
            is_single_image = True
            imgs = [imgs]

        outputs = []
        for img in imgs:
            outputs.append(self.aug(image=img)['image'])

        if is_single_image:
            outputs = outputs[0]

        return outputs

    def transform(self, results):
        """Transform function of Albumentations."""

        for k in self.keys:
            results[k] = self._apply_albu(results[k])

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(keys={self.keys}, transforms={self.transforms})'

        return repr_str
