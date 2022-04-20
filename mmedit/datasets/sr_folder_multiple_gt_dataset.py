# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os
import os.path as osp

import mmcv

from .base_sr_dataset import BaseSRDataset
from .registry import DATASETS


@DATASETS.register_module()
class SRFolderMultipleGTDataset(BaseSRDataset):
    """General dataset for video super resolution, used for recurrent networks.

    The dataset loads several LQ (Low-Quality) frames and GT (Ground-Truth)
    frames. Then it applies specified transforms and finally returns a dict
    containing paired data and other information.

    This dataset takes an annotation file specifying the sequences used in
    training or test. If no annotation file is provided, it assumes all video
    sequences under the root directory is used for training or test.

    In the annotation file (.txt), each line contains:

        1. folder name;
        2. number of frames in this sequence (in the same folder)

    Examples:

    ::

        calendar 41
        city 34
        foliage 49
        walk 47

    Args:
        lq_folder (str | :obj:`Path`): Path to a lq folder.
        gt_folder (str | :obj:`Path`): Path to a gt folder.
        pipeline (list[dict | callable]): A sequence of data transformations.
        scale (int): Upsampling scale ratio.
        ann_file (str): The path to the annotation file. If None, we assume
            that all sequences in the folder is used. Default: None
        num_input_frames (None | int): The number of frames per iteration.
            If None, the whole clip is extracted. If it is a positive integer,
            a sequence of 'num_input_frames' frames is extracted from the clip.
            Note that non-positive integers are not accepted. Default: None.
        test_mode (bool): Store `True` when building test dataset.
            Default: `True`.
    """

    def __init__(self,
                 lq_folder,
                 gt_folder,
                 pipeline,
                 scale,
                 ann_file=None,
                 num_input_frames=None,
                 test_mode=True):
        super().__init__(pipeline, scale, test_mode)

        self.lq_folder = str(lq_folder)
        self.gt_folder = str(gt_folder)
        self.ann_file = ann_file

        if num_input_frames is not None and num_input_frames <= 0:
            raise ValueError('"num_input_frames" must be None or positive, '
                             f'but got {num_input_frames}.')
        self.num_input_frames = num_input_frames

        self.data_infos = self.load_annotations()

    def _load_annotations_from_file(self):
        data_infos = []

        ann_list = mmcv.list_from_file(self.ann_file)
        for ann in ann_list:
            key, sequence_length = ann.strip().split(' ')
            if self.num_input_frames is None:
                num_input_frames = sequence_length
            else:
                num_input_frames = self.num_input_frames
            data_infos.append(
                dict(
                    lq_path=self.lq_folder,
                    gt_path=self.gt_folder,
                    key=key,
                    num_input_frames=int(num_input_frames),
                    sequence_length=int(sequence_length)))

        return data_infos

    def load_annotations(self):
        """Load annotations for the dataset.

        Returns:
            list[dict]: Returned list of dicts for paired paths of LQ and GT.
        """

        if self.ann_file:
            return self._load_annotations_from_file()

        sequences = sorted(glob.glob(osp.join(self.lq_folder, '*')))
        data_infos = []
        for sequence in sequences:
            sequence_length = len(glob.glob(osp.join(sequence, '*.png')))
            if self.num_input_frames is None:
                num_input_frames = sequence_length
            else:
                num_input_frames = self.num_input_frames
            data_infos.append(
                dict(
                    lq_path=self.lq_folder,
                    gt_path=self.gt_folder,
                    key=sequence.replace(f'{self.lq_folder}{os.sep}', ''),
                    num_input_frames=num_input_frames,
                    sequence_length=sequence_length))

        return data_infos
