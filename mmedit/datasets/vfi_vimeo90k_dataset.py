# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp

from .base_vfi_dataset import BaseVFIDataset
from .registry import DATASETS


@DATASETS.register_module()
class VFIVimeo90KDataset(BaseVFIDataset):
    """Vimeo90K dataset for video frame interpolation.

    The dataset loads two input frames and a center GT (Ground-Truth) frame.
    Then it applies specified transforms and finally returns a dict containing
    paired data and other information.

    It reads Vimeo90K keys from the txt file.
    Each line contains:

    Examples:

    ::

        00001/0389
        00001/0402

    Args:
        pipeline (list[dict | callable]): A sequence of data transformations.
        folder (str | :obj:`Path`): Path to the folder.
        ann_file (str | :obj:`Path`): Path to the annotation file.
        test_mode (bool): Store `True` when building test dataset.
            Default: `False`.
    """

    def __init__(self, pipeline, folder, ann_file, test_mode=False):
        super().__init__(pipeline, folder, ann_file, test_mode)
        self.data_infos = self.load_annotations()

    def load_annotations(self):
        """Load annotations for VimeoK dataset.

        Returns:
            list[dict]: A list of dicts for paired paths and other information.
        """
        # get keys
        with open(self.ann_file, 'r') as f:
            keys = f.read().split('\n')
            keys = [
                k.strip() for k in keys if (k.strip() is not None and k != '')
            ]

        data_infos = []
        for key in keys:
            key = key.replace('/', os.sep)
            key_folder = osp.join(self.folder, key)
            inputs_path = [
                osp.join(key_folder, 'im1.png'),
                osp.join(key_folder, 'im3.png')
            ]
            target_path = osp.join(key_folder, 'im2.png')
            data_infos.append(
                dict(
                    inputs_path=inputs_path, target_path=target_path, key=key))

        return data_infos
