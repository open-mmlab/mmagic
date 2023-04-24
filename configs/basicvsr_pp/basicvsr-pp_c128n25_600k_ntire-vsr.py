experiment_name = 'basicvsr-pp_c128n25_600k_ntire-vsr'

# model settings
model = dict(
    type='BasicVSR',
    generator=dict(
        type='BasicVSRPlusPlusNet',
        mid_channels=128,
        num_blocks=25,
        is_low_res_input=True,
        spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/'
        'basicvsr/spynet_20210409-c6c1bd09.pth',
        cpu_cache_length=100),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'),
    ensemble=dict(type='SpatialTemporalEnsemble', is_temporal_ensemble=False),
    train_cfg=dict(fix_iter=5000),
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
    ))

test_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='MirrorSequence', keys=['img', 'gt']),
    dict(type='PackInputs')
]

demo_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='MirrorSequence', keys=['img']),
    dict(type='PackInputs')
]

test_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='reds_official', task_name='vsr'),
        data_root='data/REDS',
        data_prefix=dict(img='train_sharp_bicubic/X4', gt='train_sharp'),
        ann_file='meta_info_official_val.txt',
        depth=1,
        num_input_frames=100,
        fixed_seq_len=100,
        pipeline=test_pipeline))
