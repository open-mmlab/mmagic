exp_name = 'basicvsr_plusplus_c128n25_600k_ntire_decompress_track3'

# model settings
model = dict(
    type='BasicVSR',
    generator=dict(
        type='BasicVSRPlusPlus',
        mid_channels=128,
        num_blocks=25,
        is_low_res_input=False,
        spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/'
        'basicvsr/spynet_20210409-c6c1bd09.pth',
        cpu_cache_length=100),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'),
    ensemble=dict(type='SpatialTemporalEnsemble', is_temporal_ensemble=False),
)
# model training and testing settings
train_cfg = dict(fix_iter=5000)
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=0)

# dataset settings
test_dataset_type = 'SRFolderMultipleGTDataset'
test_pipeline = [
    dict(
        type='GenerateSegmentIndices',
        interval_list=[1],
        start_idx=1,
        filename_tmpl='{:03d}.png'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(
        type='Collect',
        keys=['lq', 'gt'],
        meta_keys=['lq_path', 'gt_path', 'key'])
]

demo_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq']),
    dict(type='FramesToTensor', keys=['lq']),
    dict(type='Collect', keys=['lq'], meta_keys=['lq_path', 'key'])
]

data = dict(
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),
    test=dict(
        type=test_dataset_type,
        lq_folder='./data/NTIRE21_decompression_track3/LQ',
        gt_folder='./data/NTIRE21_decompression_track3/GT',
        pipeline=test_pipeline,
        scale=4,
        test_mode=True),
)
