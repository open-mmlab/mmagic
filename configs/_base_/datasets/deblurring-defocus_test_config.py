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

dpdd_data_root = 'data/Defocus_Deblur_Test'
dpdd_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicImageDataset',
        metainfo=dict(dataset_type='DPDD', task_name='deblurring'),
        data_root=dpdd_data_root,
        data_prefix=dict(img='source', gt='target'),
        pipeline=test_pipeline))
dpdd_evaluator = [
    dict(type='MAE', prefix='DPDD'),
    dict(type='PSNR', prefix='DPDD'),
    dict(type='SSIM', prefix='DPDD'),
]

# test config
test_cfg = dict(type='MultiTestLoop')
test_dataloader = [dpdd_dataloader]
test_evaluator = [dpdd_evaluator]
