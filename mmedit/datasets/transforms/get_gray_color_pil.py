# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np
from mmcv.transforms.base import BaseTransform

from mmedit.registry import TRANSFORMS


@TRANSFORMS.register_module()
class GenGrayColorPil(BaseTransform):

    def __init__(self, stage, keys):
        self.stage = stage
        self.keys = keys

    def transform(self, results):

        if self.stage == 'instance':
            rgb_img = results['instance']
        else:
            rgb_img = results['img']
        if len(rgb_img.shape) == 2:
            rgb_img = np.stack([rgb_img, rgb_img, rgb_img], 2)
        gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        gray_img = np.stack([gray_img, gray_img, gray_img], -1)

        results[self.keys[0]] = rgb_img
        results[self.keys[1]] = gray_img

        return results
