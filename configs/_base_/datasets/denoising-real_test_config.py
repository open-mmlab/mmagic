test_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='img',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(
        type='LoadImageFromFile',
        key='gt',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(type='PackEditInputs')
]

sidd_data_root = 'data/SIDD'
sidd_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicImageDataset',
        metainfo=dict(dataset_type='SIDD', task_name='denoising'),
        data_root=sidd_data_root,
        data_prefix=dict(img='input', gt='groundtruth'),
        pipeline=test_pipeline))
sidd_evaluator = [
    dict(type='PSNR', prefix='SIDD'),
    dict(type='SSIM', prefix='SIDD'),
]

dnd_data_root = 'data/DND'
dnd_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicImageDataset',
        metainfo=dict(dataset_type='DND', task_name='denoising'),
        data_root=dnd_data_root,
        data_prefix=dict(img='input', gt='groundtruth'),
        pipeline=test_pipeline))
dnd_evaluator = [
    dict(type='PSNR', prefix='DND'),
    dict(type='SSIM', prefix='DND'),
]

# test config
test_cfg = dict(type='MultiTestLoop')
test_dataloader = [
    sidd_dataloader,
    # dnd_dataloader,
]
test_evaluator = [
    sidd_evaluator,
    # dnd_dataloader,
]
