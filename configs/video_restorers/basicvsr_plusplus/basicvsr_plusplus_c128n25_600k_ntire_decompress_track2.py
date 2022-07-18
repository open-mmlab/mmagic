_base_ = './basicvsr_plusplus_c128n25_600k_ntire_decompress_track1.py'

experiment_name = 'basicvsr_plusplus_c128n25_600k_ntire_decompress_track2'
work_dir = f'./work_dirs/{experiment_name}'

test_dataloader = dict(
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='ntire21_track2', task_name='vsr'),
        data_root='data/NTIRE21_decompression_track2'))
