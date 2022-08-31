# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv.transforms.base import BaseTransform

from mmedit.registry import TRANSFORMS


@TRANSFORMS.register_module()
class GetMaskedImage(BaseTransform):
    """Get masked image.

    Args:
        img_key (str): Key for clean image. Default: 'gt'.
        mask_key (str): Key for mask image. The mask shape should be
            (h, w, 1) while '1' indicate holes and '0' indicate valid
            regions. Default: 'mask'.
        img_key (str): Key for output image. Default: 'img'.
        zero_value (float): Pixel value of masked area.
    """

    def __init__(self,
                 img_key='gt',
                 mask_key='mask',
                 out_key='img',
                 zero_value=127.5):
        self.img_key = img_key
        self.mask_key = mask_key
        self.out_key = out_key
        self.zero_value = zero_value

    def transform(self, results):
        """transform function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        clean_img = results[self.img_key]  # uint8
        mask = results[self.mask_key]  # uint8

        masked_img = clean_img * (1.0 - mask) + self.zero_value * mask
        masked_img = masked_img.astype(np.float32)
        results[self.out_key] = masked_img

        return results

    def __repr__(self):
        return self.__class__.__name__ + (
            f'(img_key={repr(self.img_key)}, '
            f'mask_key={repr(self.mask_key)}, '
            f'out_key={repr(self.out_key)}, '
            f'zero_value={repr(self.zero_value)})')
