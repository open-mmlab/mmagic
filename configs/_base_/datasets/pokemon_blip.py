pipeline = [
    dict(
        type='LoadImageFromHuggingFaceDataset', key='img',
        channel_order='rgb'),
    dict(type='Resize', scale=(512, 512)),
    dict(type='Flip', keys=['img'], flip_ratio=0.5, direction='horizontal'),
    dict(type='PackInputs')
]
dataset = dict(
    type='HuggingFaceDataset',
    dataset='lambdalabs/pokemon-blip-captions',
    pipeline=pipeline)
train_dataloader = dict(
    batch_size=4,
    num_workers=16,
    dataset=dataset,
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = val_evaluator = None
test_dataloader = test_evaluator = None
