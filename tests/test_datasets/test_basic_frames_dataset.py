# Copyright (c) OpenMMLab. All rights reserved.
import os
from pathlib import Path

from mmagic.datasets import BasicFramesDataset


class TestFramesDatasets:

    # TODO add a param for depth of file (for sequence_length).

    @classmethod
    def setup_class(cls):
        cls.data_root = Path(__file__).parent.parent / 'data' / 'frames'

    def test_version_1_method(self):

        # test SRREDSDataset and SRREDSMultipleGTDataset
        # need to split the data set ahead of schedule
        dataset = BasicFramesDataset(
            ann_file='ann2.txt',
            metainfo=dict(dataset_type='SRREDSDataset', task_name='vsr'),
            data_root=self.data_root,
            data_prefix=dict(
                img=f'sequence{os.sep}gt', gt=f'sequence{os.sep}gt'),
            backend_args=dict(backend='local'),
            pipeline=[],
            depth=1,
            num_input_frames=5,
            fixed_seq_len=100)
        assert dataset[0] == dict(
            key='sequence_1',
            num_input_frames=5,
            num_output_frames=None,
            sequence_length=100,
            img_path=str(self.data_root / f'sequence{os.sep}gt'),
            gt_path=str(self.data_root / f'sequence{os.sep}gt'),
            sample_idx=0)

        # test SRVimeo90KDataset and SRVimeo90KMultipleGTDataset
        # Each clip of Vimeo90K has 7 frames starting from 1.
        # So we use 9 for generating frame_index_list in Version 1.0:
        # N | frame_index_list
        # 1 | 4
        # 3 | 3,4,5
        # 5 | 2,3,4,5,6
        # 7 | 1,2,3,4,5,6,7
        # In Version 2.0, we load the list of frames directly
        dataset = BasicFramesDataset(
            ann_file='ann3.txt',
            metainfo=dict(dataset_type='SRVimeo90KDataset', task_name='vsr'),
            data_root=self.data_root,
            data_prefix=dict(img='sequence', gt='sequence'),
            pipeline=[],
            depth=2,
            load_frames_list=dict(
                img=['all'], gt=['00000000.png', '00000001.png']))
        assert dataset[0] == dict(
            key=f'gt{os.sep}sequence_1',
            num_input_frames=2,
            num_output_frames=2,
            sequence_length=None,
            img_path=[
                str(self.data_root / f'sequence{os.sep}gt' / 'sequence_1' /
                    '00000000.png'),
                str(self.data_root / f'sequence{os.sep}gt' / 'sequence_1' /
                    '00000001.png')
            ],
            gt_path=[
                str(self.data_root / f'sequence{os.sep}gt' / 'sequence_1' /
                    '00000000.png'),
                str(self.data_root / f'sequence{os.sep}gt' / 'sequence_1' /
                    '00000001.png')
            ],
            sample_idx=0)

        # test SRVid4Dataset
        dataset = BasicFramesDataset(
            ann_file='ann1.txt',
            metainfo=dict(dataset_type='vsr_folder_dataset', task_name='vsr'),
            data_root=self.data_root,
            data_prefix=dict(
                img=f'sequence{os.sep}gt', gt=f'sequence{os.sep}gt'),
            pipeline=[],
            depth=2,
            num_input_frames=2,
            num_output_frames=2,
            fixed_seq_len=3)
        assert dataset[0] == dict(
            key=f'sequence_1{os.sep}00000000',
            num_input_frames=2,
            num_output_frames=2,
            sequence_length=3,
            img_path=str(self.data_root / f'sequence{os.sep}gt'),
            gt_path=str(self.data_root / f'sequence{os.sep}gt'),
            sample_idx=0)

        # test SRTestMultipleGTDataset and SRFolderMultipleGTDataset
        dataset = BasicFramesDataset(
            ann_file='',
            metainfo=dict(dataset_type='vsr_folder_dataset', task_name='vsr'),
            data_root=self.data_root,
            data_prefix=dict(
                img=f'sequence{os.sep}gt', gt=f'sequence{os.sep}gt'),
            pipeline=[],
            num_input_frames=1,
            depth=1)
        assert dataset[0] == dict(
            key='sequence_1',
            num_input_frames=1,
            num_output_frames=None,
            sequence_length=2,
            img_path=str(self.data_root / f'sequence{os.sep}gt'),
            gt_path=str(self.data_root / f'sequence{os.sep}gt'),
            sample_idx=0)
        dataset = BasicFramesDataset(
            ann_file='ann1.txt',
            metainfo=dict(dataset_type='vsr_folder_dataset', task_name='vsr'),
            data_root=self.data_root,
            data_prefix=dict(
                img=f'sequence{os.sep}gt', gt=f'sequence{os.sep}gt'),
            pipeline=[],
            num_input_frames=1,
            depth=1)
        assert dataset[0] == dict(
            key='sequence_1',
            num_input_frames=1,
            num_output_frames=None,
            sequence_length=2,
            img_path=str(self.data_root / f'sequence{os.sep}gt'),
            gt_path=str(self.data_root / f'sequence{os.sep}gt'),
            sample_idx=0)

        # test SRFolderVideoDataset
        dataset = BasicFramesDataset(
            ann_file='',
            metainfo=dict(dataset_type='vsr_folder_dataset', task_name='vsr'),
            data_root=self.data_root,
            data_prefix=dict(
                img=f'sequence{os.sep}gt', gt=f'sequence{os.sep}gt'),
            pipeline=[],
            num_input_frames=5,
            depth=2)
        assert dataset[0] == dict(
            key=f'sequence_1{os.sep}00000000',
            num_input_frames=5,
            num_output_frames=None,
            sequence_length=2,
            img_path=str(self.data_root / f'sequence{os.sep}gt'),
            gt_path=str(self.data_root / f'sequence{os.sep}gt'),
            sample_idx=0)
        dataset = BasicFramesDataset(
            ann_file='ann2.txt',
            metainfo=dict(dataset_type='vsr_folder_dataset', task_name='vsr'),
            data_root=self.data_root,
            data_prefix=dict(
                img=f'sequence{os.sep}gt', gt=f'sequence{os.sep}gt'),
            pipeline=[],
            num_input_frames=5,
            depth=2)
        assert dataset[0] == dict(
            key=f'sequence_1{os.sep}00000000',
            num_input_frames=5,
            num_output_frames=None,
            sequence_length=2,
            img_path=str(self.data_root / f'sequence{os.sep}gt'),
            gt_path=str(self.data_root / f'sequence{os.sep}gt'),
            sample_idx=0)

        # test VFIVimeo90KDataset
        dataset = BasicFramesDataset(
            ann_file='ann3.txt',
            metainfo=dict(dataset_type='vfi_folder_dataset', task_name='vfi'),
            data_root=self.data_root,
            data_prefix=dict(img='sequence', gt='sequence'),
            filename_tmpl=dict(img='{}', gt='{}'),
            pipeline=[],
            depth=2,
            num_input_frames=2,
            num_output_frames=2,
            fixed_seq_len=3,
            load_frames_list=dict(
                img=['00000000.png', '00000001.png'], gt=['00000000.png']))
        assert dataset[0] == dict(
            key=f'gt{os.sep}sequence_1',
            num_input_frames=2,
            num_output_frames=1,
            sequence_length=None,
            img_path=[
                str(self.data_root / f'sequence{os.sep}gt' / 'sequence_1' /
                    '00000000.png'),
                str(self.data_root / f'sequence{os.sep}gt' / 'sequence_1' /
                    '00000001.png')
            ],
            gt_path=[
                str(self.data_root / f'sequence{os.sep}gt' / 'sequence_1' /
                    '00000000.png')
            ],
            sample_idx=0)

    def test_vsr_folder_dataset(self):
        # case 1: deep_path
        dataset = BasicFramesDataset(
            ann_file='',
            metainfo=dict(dataset_type='vsr_folder_dataset', task_name='vsr'),
            data_root=self.data_root,
            data_prefix=dict(
                img=f'sequence{os.sep}gt', gt=f'sequence{os.sep}gt'),
            filename_tmpl=dict(img='{}', gt='{}'),
            pipeline=[],
            depth=2,
            num_input_frames=2,
            num_output_frames=2,
            fixed_seq_len=3)

        assert dataset.data_prefix == dict(
            img=str(self.data_root / f'sequence{os.sep}gt'),
            gt=str(self.data_root / f'sequence{os.sep}gt'))
        # Serialize ``self.data_list`` to save memory
        assert dataset.data_list == []
        assert dataset[0] == dict(
            key=f'sequence_1{os.sep}00000000',
            num_input_frames=2,
            num_output_frames=2,
            sequence_length=3,
            img_path=str(self.data_root / f'sequence{os.sep}gt'),
            gt_path=str(self.data_root / f'sequence{os.sep}gt'),
            sample_idx=0)
        assert dataset[1] == dict(
            key=f'sequence_1{os.sep}00000001',
            num_input_frames=2,
            num_output_frames=2,
            sequence_length=3,
            img_path=str(self.data_root / f'sequence{os.sep}gt'),
            gt_path=str(self.data_root / f'sequence{os.sep}gt'),
            sample_idx=1)
        assert dataset[2] == dict(
            key=f'sequence_2{os.sep}00000000',
            num_input_frames=2,
            num_output_frames=2,
            sequence_length=3,
            img_path=str(self.data_root / f'sequence{os.sep}gt'),
            gt_path=str(self.data_root / f'sequence{os.sep}gt'),
            sample_idx=2)

        # case 2: not deep_path
        dataset = BasicFramesDataset(
            ann_file='',
            metainfo=dict(dataset_type='vsr_folder_dataset', task_name='vsr'),
            data_root=self.data_root,
            data_prefix=dict(
                img=f'sequence{os.sep}gt', gt=f'sequence{os.sep}gt'),
            filename_tmpl=dict(img='{}', gt='{}'),
            pipeline=[],
            depth=1,
            num_input_frames=2,
            num_output_frames=2,
            fixed_seq_len=3)

        assert dataset.data_prefix == dict(
            img=str(self.data_root / f'sequence{os.sep}gt'),
            gt=str(self.data_root / f'sequence{os.sep}gt'))
        # Serialize ``self.data_list`` to save memory
        assert dataset.data_list == []
        assert dataset[0] == dict(
            key='sequence_1',
            num_input_frames=2,
            num_output_frames=2,
            sequence_length=3,
            img_path=str(self.data_root / f'sequence{os.sep}gt'),
            gt_path=str(self.data_root / f'sequence{os.sep}gt'),
            sample_idx=0)
        assert dataset[1] == dict(
            key='sequence_2',
            num_input_frames=2,
            num_output_frames=2,
            sequence_length=3,
            img_path=str(self.data_root / f'sequence{os.sep}gt'),
            gt_path=str(self.data_root / f'sequence{os.sep}gt'),
            sample_idx=1)

        # case 3: no fixed_seq_len
        dataset = BasicFramesDataset(
            ann_file='',
            metainfo=dict(dataset_type='vsr_folder_dataset', task_name='vsr'),
            data_root=self.data_root,
            data_prefix=dict(
                img=f'sequence{os.sep}gt', gt=f'sequence{os.sep}gt'),
            filename_tmpl=dict(img='{}', gt='{}'),
            pipeline=[],
            depth=1,
            num_input_frames=2,
            num_output_frames=2,
            fixed_seq_len=None)

        assert dataset.data_prefix == dict(
            img=str(self.data_root / f'sequence{os.sep}gt'),
            gt=str(self.data_root / f'sequence{os.sep}gt'))
        # Serialize ``self.data_list`` to save memory
        assert dataset.data_list == []
        assert dataset[0] == dict(
            key='sequence_1',
            num_input_frames=2,
            num_output_frames=2,
            sequence_length=2,
            img_path=str(self.data_root / f'sequence{os.sep}gt'),
            gt_path=str(self.data_root / f'sequence{os.sep}gt'),
            sample_idx=0)
        assert dataset[1] == dict(
            key='sequence_2',
            num_input_frames=2,
            num_output_frames=2,
            sequence_length=3,
            img_path=str(self.data_root / f'sequence{os.sep}gt'),
            gt_path=str(self.data_root / f'sequence{os.sep}gt'),
            sample_idx=1)

        # case 4: not deep_path, load_frames_list
        dataset = BasicFramesDataset(
            ann_file='',
            metainfo=dict(dataset_type='vsr_folder_dataset', task_name='vsr'),
            data_root=self.data_root,
            data_prefix=dict(
                img=f'sequence{os.sep}gt', gt=f'sequence{os.sep}gt'),
            filename_tmpl=dict(img='{}', gt='{}'),
            pipeline=[],
            depth=1,
            num_input_frames=2,
            num_output_frames=2,
            fixed_seq_len=3,
            load_frames_list=dict(img=['all'], gt=['00000000.png']))

        assert dataset.data_prefix == dict(
            img=str(self.data_root / f'sequence{os.sep}gt'),
            gt=str(self.data_root / f'sequence{os.sep}gt'))
        # Serialize ``self.data_list`` to save memory
        assert dataset.data_list == []
        assert dataset[0] == dict(
            key='sequence_1',
            num_input_frames=2,
            num_output_frames=1,
            sequence_length=None,
            img_path=[
                str(self.data_root / f'sequence{os.sep}gt' / 'sequence_1' /
                    '00000000.png'),
                str(self.data_root / f'sequence{os.sep}gt' / 'sequence_1' /
                    '00000001.png')
            ],
            gt_path=[
                str(self.data_root / f'sequence{os.sep}gt' / 'sequence_1' /
                    '00000000.png')
            ],
            sample_idx=0)
        assert dataset[1] == dict(
            key='sequence_2',
            num_input_frames=3,
            num_output_frames=1,
            sequence_length=None,
            img_path=[
                str(self.data_root / f'sequence{os.sep}gt' / 'sequence_2' /
                    '00000000.png'),
                str(self.data_root / f'sequence{os.sep}gt' / 'sequence_2' /
                    '00000001.png'),
                str(self.data_root / f'sequence{os.sep}gt' / 'sequence_2' /
                    '00000002.png')
            ],
            gt_path=[
                str(self.data_root / f'sequence{os.sep}gt' / 'sequence_2' /
                    '00000000.png')
            ],
            sample_idx=1)

        # case 5: deep_path, load_frames_list
        dataset = BasicFramesDataset(
            ann_file='',
            metainfo=dict(dataset_type='vsr_folder_dataset', task_name='vsr'),
            data_root=self.data_root,
            data_prefix=dict(img='sequence', gt='sequence'),
            filename_tmpl=dict(img='{}', gt='{}'),
            pipeline=[],
            depth=2,
            num_input_frames=2,
            num_output_frames=2,
            fixed_seq_len=3,
            load_frames_list=dict(img=['00000000.png'], gt=['all']))

        assert dataset.data_prefix == dict(
            img=str(self.data_root / 'sequence'),
            gt=str(self.data_root / 'sequence'))
        # Serialize ``self.data_list`` to save memory
        assert dataset.data_list == []
        assert dataset[0] == dict(
            key=f'gt{os.sep}sequence_1',
            num_input_frames=1,
            num_output_frames=2,
            sequence_length=None,
            img_path=[
                str(self.data_root / f'sequence{os.sep}gt' / 'sequence_1' /
                    '00000000.png')
            ],
            gt_path=[
                str(self.data_root / f'sequence{os.sep}gt' / 'sequence_1' /
                    '00000000.png'),
                str(self.data_root / f'sequence{os.sep}gt' / 'sequence_1' /
                    '00000001.png')
            ],
            sample_idx=0)
        assert dataset[1] == dict(
            key=f'gt{os.sep}sequence_2',
            num_input_frames=1,
            num_output_frames=3,
            sequence_length=None,
            img_path=[
                str(self.data_root / f'sequence{os.sep}gt' / 'sequence_2' /
                    '00000000.png')
            ],
            gt_path=[
                str(self.data_root / f'sequence{os.sep}gt' / 'sequence_2' /
                    '00000000.png'),
                str(self.data_root / f'sequence{os.sep}gt' / 'sequence_2' /
                    '00000001.png'),
                str(self.data_root / f'sequence{os.sep}gt' / 'sequence_2' /
                    '00000002.png')
            ],
            sample_idx=1)

    def test_vsr_ann_dataset(self):
        # case 1: deep_path, not deep_ann
        dataset = BasicFramesDataset(
            ann_file='ann1.txt',
            metainfo=dict(dataset_type='vsr_folder_dataset', task_name='vsr'),
            data_root=self.data_root,
            data_prefix=dict(
                img=f'sequence{os.sep}gt', gt=f'sequence{os.sep}gt'),
            filename_tmpl=dict(img='{}', gt='{}'),
            pipeline=[],
            depth=2,
            num_input_frames=2,
            num_output_frames=2,
            fixed_seq_len=3)

        assert dataset.data_prefix == dict(
            img=str(self.data_root / f'sequence{os.sep}gt'),
            gt=str(self.data_root / f'sequence{os.sep}gt'))
        assert dataset.ann_file == str(self.data_root / 'ann1.txt')
        # Serialize ``self.data_list`` to save memory
        assert dataset.data_list == []
        assert dataset[0] == dict(
            key=f'sequence_1{os.sep}00000000',
            num_input_frames=2,
            num_output_frames=2,
            sequence_length=3,
            img_path=str(self.data_root / f'sequence{os.sep}gt'),
            gt_path=str(self.data_root / f'sequence{os.sep}gt'),
            sample_idx=0)
        assert dataset[1] == dict(
            key=f'sequence_1{os.sep}00000001',
            num_input_frames=2,
            num_output_frames=2,
            sequence_length=3,
            img_path=str(self.data_root / f'sequence{os.sep}gt'),
            gt_path=str(self.data_root / f'sequence{os.sep}gt'),
            sample_idx=1)
        assert dataset[2] == dict(
            key=f'sequence_2{os.sep}00000000',
            num_input_frames=2,
            num_output_frames=2,
            sequence_length=3,
            img_path=str(self.data_root / f'sequence{os.sep}gt'),
            gt_path=str(self.data_root / f'sequence{os.sep}gt'),
            sample_idx=2)

        # case 2: deep_path, deep_ann
        dataset = BasicFramesDataset(
            ann_file='ann2.txt',
            metainfo=dict(dataset_type='vsr_folder_dataset', task_name='vsr'),
            data_root=self.data_root,
            data_prefix=dict(
                img=f'sequence{os.sep}gt', gt=f'sequence{os.sep}gt'),
            filename_tmpl=dict(img='{}', gt='{}'),
            pipeline=[],
            depth=2,
            num_input_frames=2,
            num_output_frames=2,
            fixed_seq_len=3)

        assert dataset.data_prefix == dict(
            img=str(self.data_root / f'sequence{os.sep}gt'),
            gt=str(self.data_root / f'sequence{os.sep}gt'))
        assert dataset.ann_file == str(self.data_root / 'ann2.txt')
        # Serialize ``self.data_list`` to save memory
        assert dataset.data_list == []
        assert dataset[0] == dict(
            key=f'sequence_1{os.sep}00000000',
            num_input_frames=2,
            num_output_frames=2,
            sequence_length=3,
            img_path=str(self.data_root / f'sequence{os.sep}gt'),
            gt_path=str(self.data_root / f'sequence{os.sep}gt'),
            sample_idx=0)
        assert dataset[1] == dict(
            key=f'sequence_1{os.sep}00000001',
            num_input_frames=2,
            num_output_frames=2,
            sequence_length=3,
            img_path=str(self.data_root / f'sequence{os.sep}gt'),
            gt_path=str(self.data_root / f'sequence{os.sep}gt'),
            sample_idx=1)
        assert dataset[2] == dict(
            key=f'sequence_2{os.sep}00000000',
            num_input_frames=2,
            num_output_frames=2,
            sequence_length=3,
            img_path=str(self.data_root / f'sequence{os.sep}gt'),
            gt_path=str(self.data_root / f'sequence{os.sep}gt'),
            sample_idx=2)

        # case 3: not deep_path, not deep_ann
        dataset = BasicFramesDataset(
            ann_file='ann1.txt',
            metainfo=dict(dataset_type='vsr_folder_dataset', task_name='vsr'),
            data_root=self.data_root,
            data_prefix=dict(
                img=f'sequence{os.sep}gt', gt=f'sequence{os.sep}gt'),
            filename_tmpl=dict(img='{}', gt='{}'),
            pipeline=[],
            depth=1,
            num_input_frames=2,
            num_output_frames=2,
            fixed_seq_len=3)

        assert dataset.data_prefix == dict(
            img=str(self.data_root / f'sequence{os.sep}gt'),
            gt=str(self.data_root / f'sequence{os.sep}gt'))
        assert dataset.ann_file == str(self.data_root / 'ann1.txt')
        # Serialize ``self.data_list`` to save memory
        assert dataset.data_list == []
        assert dataset[0] == dict(
            key='sequence_1',
            num_input_frames=2,
            num_output_frames=2,
            sequence_length=3,
            img_path=str(self.data_root / f'sequence{os.sep}gt'),
            gt_path=str(self.data_root / f'sequence{os.sep}gt'),
            sample_idx=0)
        assert dataset[1] == dict(
            key='sequence_2',
            num_input_frames=2,
            num_output_frames=2,
            sequence_length=3,
            img_path=str(self.data_root / f'sequence{os.sep}gt'),
            gt_path=str(self.data_root / f'sequence{os.sep}gt'),
            sample_idx=1)

        # case 4: not deep_path, deep_ann
        dataset = BasicFramesDataset(
            ann_file='ann2.txt',
            metainfo=dict(dataset_type='vsr_folder_dataset', task_name='vsr'),
            data_root=self.data_root,
            data_prefix=dict(
                img=f'sequence{os.sep}gt', gt=f'sequence{os.sep}gt'),
            filename_tmpl=dict(img='{}', gt='{}'),
            pipeline=[],
            depth=1,
            num_input_frames=2,
            num_output_frames=2,
            fixed_seq_len=3)

        assert dataset.data_prefix == dict(
            img=str(self.data_root / f'sequence{os.sep}gt'),
            gt=str(self.data_root / f'sequence{os.sep}gt'))
        assert dataset.ann_file == str(self.data_root / 'ann2.txt')
        # Serialize ``self.data_list`` to save memory
        assert dataset.data_list == []
        assert dataset[0] == dict(
            key='sequence_1',
            num_input_frames=2,
            num_output_frames=2,
            sequence_length=3,
            img_path=str(self.data_root / f'sequence{os.sep}gt'),
            gt_path=str(self.data_root / f'sequence{os.sep}gt'),
            sample_idx=0)
        assert dataset[1] == dict(
            key='sequence_2',
            num_input_frames=2,
            num_output_frames=2,
            sequence_length=3,
            img_path=str(self.data_root / f'sequence{os.sep}gt'),
            gt_path=str(self.data_root / f'sequence{os.sep}gt'),
            sample_idx=1)

    def test_vfi_ann_dataset(self):
        # case 1: not deep_path
        dataset = BasicFramesDataset(
            ann_file='ann1.txt',
            metainfo=dict(dataset_type='vfi_folder_dataset', task_name='vfi'),
            data_root=self.data_root,
            data_prefix=dict(
                img=f'sequence{os.sep}gt', gt=f'sequence{os.sep}gt'),
            filename_tmpl=dict(img='{}', gt='{}'),
            pipeline=[],
            depth=1,
            num_input_frames=2,
            num_output_frames=2,
            fixed_seq_len=3,
            load_frames_list=dict(
                img=['00000000.png', '00000001.png'], gt=['00000000.png']))

        assert dataset.data_prefix == dict(
            img=str(self.data_root / f'sequence{os.sep}gt'),
            gt=str(self.data_root / f'sequence{os.sep}gt'))
        assert dataset.ann_file == str(self.data_root / 'ann1.txt')
        # Serialize ``self.data_list`` to save memory
        assert dataset.data_list == []
        assert dataset[0] == dict(
            key='sequence_1',
            num_input_frames=2,
            num_output_frames=1,
            sequence_length=None,
            img_path=[
                str(self.data_root / f'sequence{os.sep}gt' / 'sequence_1' /
                    '00000000.png'),
                str(self.data_root / f'sequence{os.sep}gt' / 'sequence_1' /
                    '00000001.png')
            ],
            gt_path=[
                str(self.data_root / f'sequence{os.sep}gt' / 'sequence_1' /
                    '00000000.png')
            ],
            sample_idx=0)
        assert dataset[1] == dict(
            key='sequence_2',
            num_input_frames=2,
            num_output_frames=1,
            sequence_length=None,
            img_path=[
                str(self.data_root / f'sequence{os.sep}gt' / 'sequence_2' /
                    '00000000.png'),
                str(self.data_root / f'sequence{os.sep}gt' / 'sequence_2' /
                    '00000001.png')
            ],
            gt_path=[
                str(self.data_root / f'sequence{os.sep}gt' / 'sequence_2' /
                    '00000000.png')
            ],
            sample_idx=1)

        # case 2: deep_path
        dataset = BasicFramesDataset(
            ann_file='ann3.txt',
            metainfo=dict(dataset_type='vfi_folder_dataset', task_name='vfi'),
            data_root=self.data_root,
            data_prefix=dict(img='sequence', gt='sequence'),
            filename_tmpl=dict(img='{}', gt='{}'),
            pipeline=[],
            depth=2,
            num_input_frames=2,
            num_output_frames=2,
            fixed_seq_len=3,
            load_frames_list=dict(
                img=['00000000.png', '00000001.png'], gt=['00000000.png']))

        assert dataset.data_prefix == dict(
            img=str(self.data_root / 'sequence'),
            gt=str(self.data_root / 'sequence'))
        assert dataset.ann_file == str(self.data_root / 'ann3.txt')
        # Serialize ``self.data_list`` to save memory
        assert dataset.data_list == []
        assert dataset[0] == dict(
            key=f'gt{os.sep}sequence_1',
            num_input_frames=2,
            num_output_frames=1,
            sequence_length=None,
            img_path=[
                str(self.data_root / f'sequence{os.sep}gt' / 'sequence_1' /
                    '00000000.png'),
                str(self.data_root / f'sequence{os.sep}gt' / 'sequence_1' /
                    '00000001.png')
            ],
            gt_path=[
                str(self.data_root / f'sequence{os.sep}gt' / 'sequence_1' /
                    '00000000.png')
            ],
            sample_idx=0)
        assert dataset[1] == dict(
            key=f'gt{os.sep}sequence_2',
            num_input_frames=2,
            num_output_frames=1,
            sequence_length=None,
            img_path=[
                str(self.data_root / f'sequence{os.sep}gt' / 'sequence_2' /
                    '00000000.png'),
                str(self.data_root / f'sequence{os.sep}gt' / 'sequence_2' /
                    '00000001.png')
            ],
            gt_path=[
                str(self.data_root / f'sequence{os.sep}gt' / 'sequence_2' /
                    '00000000.png')
            ],
            sample_idx=1)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
