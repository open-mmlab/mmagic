# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp

from .base_sr_dataset import BaseSRDataset
from .registry import DATASETS


@DATASETS.register_module()
class SRVimeo90KDataset(BaseSRDataset):
    """Vimeo90K dataset for video super resolution.

    The dataset loads several LQ (Low-Quality) frames and a center GT
    (Ground-Truth) frame. Then it applies specified transforms and finally
    returns a dict containing paired data and other information.

    It reads Vimeo90K keys from the txt file.
    Each line contains:
    1. image name; 2, image shape, separated by a white space.
    Examples:

    ::

        00001/0266 (256, 448, 3)
        00001/0268 (256, 448, 3)

    Args:
        lq_folder (str | :obj:`Path`): Path to a lq folder.
        gt_folder (str | :obj:`Path`): Path to a gt folder.
        ann_file (str | :obj:`Path`): Path to the annotation file.
        num_input_frames (int): Window size for input frames.
        pipeline (list[dict | callable]): A sequence of data transformations.
        scale (int): Upsampling scale ratio.
        test_mode (bool): Store `True` when building test dataset.
            Default: `False`.
    """

    def __init__(self,
                 lq_folder,
                 gt_folder,
                 ann_file,
                 num_input_frames,
                 pipeline,
                 scale,
                 test_mode=False):
        super().__init__(pipeline, scale, test_mode)
        assert num_input_frames % 2 == 1, (
            f'num_input_frames should be odd numbers, '
            f'but received {num_input_frames}.')
        self.lq_folder = str(lq_folder)
        self.gt_folder = str(gt_folder)
        self.ann_file = str(ann_file)
        self.num_input_frames = num_input_frames
        self.data_infos = self.load_annotations()

    def load_annotations(self):
        """Load annotations for VimeoK dataset.

        Returns:
            list[dict]: A list of dicts for paired paths and other information.
        """
        # get keys
        with open(self.ann_file, 'r') as fin:
            keys = [line.strip().split(' ')[0] for line in fin]
        # get frame index list for LQ frames
        frame_index_list = []
        for i in range(self.num_input_frames):
            # Each clip of Vimeo90K has 7 frames starting from 1. So we use 9
            # for generating frame_index_list:
            # N | frame_index_list
            # 1 | 4
            # 3 | 3,4,5
            # 5 | 2,3,4,5,6
            # 7 | 1,2,3,4,5,6,7
            frame_index_list.append(i + (9 - self.num_input_frames) // 2)

        data_infos = []
        for key in keys:
            key = key.replace('/', os.sep)
            folder, subfolder = key.split(os.sep)
            lq_paths = []
            for i in frame_index_list:
                lq_paths.append(
                    osp.join(self.lq_folder, folder, subfolder, f'im{i}.png'))
            gt_paths = [osp.join(self.gt_folder, folder, subfolder, 'im4.png')]

            data_infos.append(
                dict(lq_path=lq_paths, gt_path=gt_paths, key=key))

        return data_infos
