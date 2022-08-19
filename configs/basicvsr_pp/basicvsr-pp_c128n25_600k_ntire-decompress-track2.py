_base_ = './basicvsr-pp_c128n25_ntire-decompress-track1_600k.py'

experiment_name = 'basicvsr-pp_c128n25_ntire-decompress-track2_600k'
work_dir = f'./work_dirs/{experiment_name}'

test_dataloader = dict(
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='ntire21_track2', task_name='vsr'),
        data_root='data/NTIRE21_decompression_track2'))
