### Use intermediate variables in configs

Some intermediate variables are used in the configs files, like `train_pipeline`/`test_pipeline` in datasets.

For example, we would like to first define the `train_pipeline`/`test_pipeline` and pass them into `train_dataloader`/`test_dataloader`. Thus, `train_pipeline`/`test_pipeline` are intermediate variable.

```python
...
train_pipeline = [
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
    dict(type='SetValues', dictionary=dict(scale=scale)),
    dict(type='PairedRandomCrop', gt_patch_size=128),
    dict(
        type='Flip',
        keys=['img', 'gt'],
        flip_ratio=0.5,
        direction='horizontal'),
    dict(
        type='Flip', keys=['img', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['img', 'gt'], transpose_ratio=0.5),
    dict(type='ToTensor', keys=['img', 'gt']),
    dict(type='PackEditInputs')
]
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
    dict(type='ToTensor', keys=['img', 'gt']),
    dict(type='PackEditInputs')
]

# dataset settings
dataset_type = 'BasicImageDataset'
data_root = 'data'

train_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file='meta_info_DIV2K800sub_GT.txt',
        metainfo=dict(dataset_type='div2k', task_name='sisr'),
        data_root=data_root + '/DIV2K',
        data_prefix=dict(
            img='DIV2K_train_LR_bicubic/X4_sub', gt='DIV2K_train_HR_sub'),
        filename_tmpl=dict(img='{}', gt='{}'),
        pipeline=train_pipeline))

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
...
```
