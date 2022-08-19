_base_ = './basicvsr-pp_c128n25_600k_ntire-decompress-track1.py'

experiment_name = 'basicvsr-pp_c128n25_600k_ntire-decompress-track3'
work_dir = f'./work_dirs/{experiment_name}'

test_dataloader = dict(
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='ntire21_track3', task_name='vsr'),
        data_root='data/NTIRE21_decompression_track3'))
