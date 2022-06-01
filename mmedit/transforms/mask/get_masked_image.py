# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.transforms.base import BaseTransform

from mmedit.registry import TRANSFORMS


@TRANSFORMS.register_module()
class GetMaskedImage(BaseTransform):
    """Get masked image.

    Args:
        img_name (str): Key for clean image.
        mask_name (str): Key for mask image. The mask shape should be
            (h, w, 1) while '1' indicate holes and '0' indicate valid
            regions.
    """

    def __init__(self, img_name='gt_img', mask_name='mask'):
        self.img_name = img_name
        self.mask_name = mask_name

    def transform(self, results):
        """transform function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        clean_img = results[self.img_name]
        mask = results[self.mask_name]

        masked_img = clean_img * (1. - mask)
        results['masked_img'] = masked_img

        return results

    def __repr__(self):
        return self.__class__.__name__ + (
            f"(img_name='{self.img_name}', mask_name='{self.mask_name}')")
