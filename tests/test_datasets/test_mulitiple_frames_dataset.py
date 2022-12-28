# Copyright (c) OpenMMLab. All rights reserved.
import os
from pathlib import Path

from mmedit.datasets import MultipleFramesDataset


class TestMultipleFramesDatasets:

    @classmethod
    def setup_class(cls):
        cls.data_root = Path(__file__).parent.parent / 'data' / 'frames'

    def test_multiple_method(self):

        # test num_input_frames
        dataset = MultipleFramesDataset(
            metainfo=dict(dataset_type='SRREDSDataset', task_name='vsr'),
            data_root=self.data_root,
            data_prefix=dict(
                img=f'sequence{os.sep}gt', gt=f'sequence{os.sep}gt'),
            pipeline=[],
            depth=1,
            load_frames_list=dict(img=[0, 1], gt=[0]))
        assert dataset.__len__() == 3
        assert dataset[0] == dict(
            key='sequence_1_00000000',
            num_input_frames=2,
            num_output_frames=1,
            num_frame_this_data=2,
            sequence_length=None,
            img_path=[
                str(self.data_root /
                    f'sequence{os.sep}gt/sequence_1/00000000.png'),
                str(self.data_root /
                    f'sequence{os.sep}gt/sequence_1/00000001.png')
            ],
            gt_path=[
                str(self.data_root /
                    f'sequence{os.sep}gt/sequence_1/00000000.png')
            ],
            sample_idx=0)

        # test num_input_frames
        dataset = MultipleFramesDataset(
            metainfo=dict(dataset_type='SRREDSDataset', task_name='vsr'),
            data_root=self.data_root,
            data_prefix=dict(
                img=f'sequence{os.sep}gt', gt=f'sequence{os.sep}gt'),
            pipeline=[],
            depth=1,
            num_input_frames=2)
        assert dataset.__len__() == 3
        assert dataset[0] == dict(
            key='sequence_1_00000000',
            num_input_frames=2,
            num_output_frames=2,
            num_frame_this_data=2,
            sequence_length=None,
            img_path=[
                str(self.data_root /
                    f'sequence{os.sep}gt/sequence_1/00000000.png'),
                str(self.data_root /
                    f'sequence{os.sep}gt/sequence_1/00000001.png')
            ],
            gt_path=[
                str(self.data_root /
                    f'sequence{os.sep}gt/sequence_1/00000000.png'),
                str(self.data_root /
                    f'sequence{os.sep}gt/sequence_1/00000001.png')
            ],
            sample_idx=0)
