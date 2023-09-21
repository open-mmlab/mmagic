pipeline = [
    dict(
        type='LoadImageFromHuggingFaceDataset', key='img',
        channel_order='rgb'),
    dict(type='ResizeEdge', scale=1024),
    dict(type='RandomCropXL', size=1024),
    dict(type='FlipXL', keys=['img'], flip_ratio=0.5, direction='horizontal'),
    dict(type='ComputeTimeIds'),
    dict(type='PackInputs', keys=['merged', 'img', 'time_ids']),
]
dataset = dict(
    type='HuggingFaceDataset',
    dataset='lambdalabs/pokemon-blip-captions',
    pipeline=pipeline)
train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dataset,
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = val_evaluator = None
test_dataloader = test_evaluator = None
