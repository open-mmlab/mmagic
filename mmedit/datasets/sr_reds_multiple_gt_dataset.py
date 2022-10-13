# Copyright (c) OpenMMLab. All rights reserved.
from .base_sr_dataset import BaseSRDataset
from .registry import DATASETS


@DATASETS.register_module()
class SRREDSMultipleGTDataset(BaseSRDataset):
    """REDS dataset for video super resolution for recurrent networks.

    The dataset loads several LQ (Low-Quality) frames and GT (Ground-Truth)
    frames. Then it applies specified transforms and finally returns a dict
    containing paired data and other information.

    Args:
        lq_folder (str | :obj:`Path`): Path to a lq folder.
        gt_folder (str | :obj:`Path`): Path to a gt folder.
        num_input_frames (int): Number of input frames.
        pipeline (list[dict | callable]): A sequence of data transformations.
        scale (int): Upsampling scale ratio.
        val_partition (str): Validation partition mode. Choices ['official' or
        'REDS4']. Default: 'official'.
        repeat (int): Number of replication of the validation set. This is used
            to allow training REDS4 with more than 4 GPUs. For example, if
            8 GPUs are used, this number can be set to 2. Default: 1.
        test_mode (bool): Store `True` when building test dataset.
            Default: `False`.
    """

    def __init__(self,
                 lq_folder,
                 gt_folder,
                 num_input_frames,
                 pipeline,
                 scale,
                 val_partition='official',
                 repeat=1,
                 test_mode=False):

        self.repeat = repeat
        if not isinstance(repeat, int):
            raise TypeError('"repeat" must be an integer, but got '
                            f'{type(repeat)}.')

        super().__init__(pipeline, scale, test_mode)
        self.lq_folder = str(lq_folder)
        self.gt_folder = str(gt_folder)
        self.num_input_frames = num_input_frames
        self.val_partition = val_partition
        self.data_infos = self.load_annotations()

    def load_annotations(self):
        """Load annotations for REDS dataset.

        Returns:
            list[dict]: A list of dicts for paired paths and other information.
        """
        # generate keys
        keys = [f'{i:03d}' for i in range(0, 270)]

        if self.val_partition == 'REDS4':
            val_partition = ['000', '011', '015', '020']
        elif self.val_partition == 'official':
            val_partition = [f'{i:03d}' for i in range(240, 270)]
        else:
            raise ValueError(
                f'Wrong validation partition {self.val_partition}.'
                f'Supported ones are ["official", "REDS4"]')

        if self.test_mode:
            keys = [v for v in keys if v in val_partition]
            keys *= self.repeat
        else:
            keys = [v for v in keys if v not in val_partition]

        data_infos = []
        for key in keys:
            data_infos.append(
                dict(
                    lq_path=self.lq_folder,
                    gt_path=self.gt_folder,
                    key=key,
                    sequence_length=100,  # REDS has 100 frames for each clip
                    num_input_frames=self.num_input_frames))

        return data_infos
