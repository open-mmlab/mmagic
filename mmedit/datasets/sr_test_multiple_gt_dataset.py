# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os
import os.path as osp
import warnings

from .base_sr_dataset import BaseSRDataset
from .registry import DATASETS


@DATASETS.register_module()
class SRTestMultipleGTDataset(BaseSRDataset):
    """Test dataset for video super resolution for recurrent networks.

    It assumes all video sequences under the root directory is used for test.

    The dataset loads several LQ (Low-Quality) frames and GT (Ground-Truth)
    frames. Then it applies specified transforms and finally returns a dict
    containing paired data and other information.

    Args:
        lq_folder (str | :obj:`Path`): Path to a lq folder.
        gt_folder (str | :obj:`Path`): Path to a gt folder.
        pipeline (list[dict | callable]): A sequence of data transformations.
        scale (int): Upsampling scale ratio.
        test_mode (bool): Store `True` when building test dataset.
            Default: `True`.
    """

    def __init__(self, lq_folder, gt_folder, pipeline, scale, test_mode=True):
        super().__init__(pipeline, scale, test_mode)

        warnings.warn('"SRTestMultipleGTDataset" have been deprecated and '
                      'will be removed in future release. Please use '
                      '"SRFolderMultipleGTDataset" instead. Details see '
                      'https://github.com/open-mmlab/mmediting/pull/355')

        self.lq_folder = str(lq_folder)
        self.gt_folder = str(gt_folder)
        self.data_infos = self.load_annotations()

    def load_annotations(self):
        """Load annoations for the test dataset.

        Returns:
            list[dict]: A list of dicts for paired paths and other information.
        """

        sequences = sorted(glob.glob(osp.join(self.lq_folder, '*')))

        data_infos = []
        for sequence in sequences:
            sequence_length = len(glob.glob(osp.join(sequence, '*.png')))
            data_infos.append(
                dict(
                    lq_path=self.lq_folder,
                    gt_path=self.gt_folder,
                    key=sequence.replace(f'{self.lq_folder}{os.sep}', ''),
                    sequence_length=int(sequence_length)))

        return data_infos
