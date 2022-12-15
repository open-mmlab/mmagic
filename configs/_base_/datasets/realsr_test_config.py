test_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='img',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(type='PackEditInputs')
]

# test config for RealSRSet+5images
RealSRSet_data_root = 'data'
RealSRSet_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicImageDataset',
        metainfo=dict(dataset_type='realsrset', task_name='realsr'),
        data_root=RealSRSet_data_root,
        data_prefix=dict(img='RealSRSet+5images', gt='RealSRSet+5images'),
        pipeline=test_pipeline))
RealSRSet_evaluator = []

# test config
test_cfg = dict(type='MultiTestLoop')
test_dataloader = [RealSRSet_dataloader]
test_evaluator = [RealSRSet_evaluator]
