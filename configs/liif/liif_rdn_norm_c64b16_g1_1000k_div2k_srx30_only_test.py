_base_ = './liif_rdn_norm_c64b16_g1_1000k_div2k.py'

experiment_name = 'liif_rdn_norm_x2-4_c64b16_g1_1000k_div2k_srx30'
work_dir = f'./work_dirs/{experiment_name}'

scale_test = 30
data_root = 'data'
dataset_type = 'BasicImageDataset'
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='gt',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(
        type='LoadImageFromFile',
        key='img',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(type='ToTensor', keys=['img', 'gt']),
    dict(type='GenerateCoordinateAndCell', scale=scale_test, reshape_gt=False),
    dict(type='PackEditInputs')
]

test_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(dataset_type='set5', task_name='sisr'),
        data_root=data_root + '/Set5',
        data_prefix=dict(img='LRbicx4', gt='GTmod12'),
        pipeline=test_pipeline))
