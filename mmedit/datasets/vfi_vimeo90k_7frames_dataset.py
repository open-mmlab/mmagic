# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp

from .base_vfi_dataset import BaseVFIDataset
from .registry import DATASETS


@DATASETS.register_module()
class VFIVimeo90K7FramesDataset(BaseVFIDataset):
    """Utilize Vimeo90K dataset (7 frames) for video frame interpolation.

    Load 7 GT (Ground-Truth) frames from the dataset, predict several frame(s)
    from other frames.
    Then it applies specified transforms and finally returns a dict
    containing paired data and other information.

    It reads Vimeo90K keys from the txt file. Each line contains:

        1. video frame folder
        2. number of frames
        3. image shape

    Examples:

    ::

        00001/0266 7 (256,448,3)
        00001/0268 7 (256,448,3)

    Note: Only `video frame folder` is required information.

    Args:
        folder (str | :obj:`Path`): Path to image folder.
        ann_file (str | :obj:`Path`): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transformations.
        input_frames (list[int]): Index of input frames.
        target_frames (list[int]): Index of target frames.
        test_mode (bool): Store `True` when building test dataset.
            Default: `False`.
    """

    def __init__(self,
                 folder,
                 ann_file,
                 pipeline,
                 input_frames,
                 target_frames,
                 test_mode=False):
        super().__init__(
            pipeline=pipeline,
            folder=folder,
            ann_file=ann_file,
            test_mode=test_mode)

        self.input_frames = input_frames
        self.target_frames = target_frames

        self.data_infos = self.load_annotations()

    def load_annotations(self):
        """Load annoations for Vimeo-90K dataset.

        Returns:
            list[dict]: A list of dicts for paired paths and other information.
        """
        # get keys
        with open(self.ann_file, 'r') as fin:
            keys = [line.strip().split(' ')[0] for line in fin]

        data_infos = []
        for key in keys:
            key = key.replace('/', os.sep)
            inputs_path = [
                osp.join(self.folder, key, f'im{i}.png')
                for i in self.input_frames
            ]
            target_path = [
                osp.join(self.folder, key, f'im{i}.png')
                for i in self.target_frames
            ]

            data_infos.append(
                dict(
                    inputs_path=inputs_path, target_path=target_path, key=key))

        return data_infos
