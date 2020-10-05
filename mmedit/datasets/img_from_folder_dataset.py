import os.path as osp
import random

import mmcv

from .base_dataset import BaseDataset
from .registry import DATASETS


@DATASETS.register_module()
class ImgFromFolderDataset(BaseDataset):
    """Image dataset for inpainting.
    """

    def __init__(self,
                 imgs_dir,
                 pipeline,
                 img_suffix=('.jpg', '.png'),
                 num_samples=-1,
                 test_mode=False):
        super(ImgFromFolderDataset, self).__init__(pipeline, test_mode)
        self.imgs_dir = str(imgs_dir)
        self.img_suffix = img_suffix
        self.num_samples = num_samples
        self.data_infos = self.load_annotations()

    def load_annotations(self):
        """Load annotations for dataset.

        Returns:
            list[dict]: Contain dataset annotations.
        """

        imgs = list(
            mmcv.scandir(
                self.imgs_dir, suffix=self.img_suffix, recursive=True))

        # random sample a subset
        if self.num_samples > 0:
            imgs = random.sample(imgs, self.num_samples)

        imgs = [osp.join(self.imgs_dir, x) for x in imgs]
        img_infos = [dict(img_path=x) for x in imgs]

        return img_infos
