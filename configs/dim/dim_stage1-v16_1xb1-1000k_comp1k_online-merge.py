_base_ = ['./dim_stage1-v16_1xb1-1000k_comp1k.py']
save_dir = './work_dirs/'
experiment_name = 'dim_stage1-v16_1xb1-1000k_comp1k_online-merge'

# dataset settings
train_pipeline = [
    dict(type='LoadImageFromFile', key='alpha', color_type='grayscale'),
    dict(type='LoadImageFromFile', key='fg'),
    dict(type='LoadImageFromFile', key='bg'),
    dict(type='MergeFgAndBg'),
    dict(
        type='CropAroundUnknown',
        keys=['alpha', 'merged', 'fg', 'bg'],
        crop_sizes=[320, 480, 640]),
    dict(type='Flip', keys=['alpha', 'merged', 'fg', 'bg']),
    dict(
        type='Resize',
        keys=['alpha', 'merged', 'fg', 'bg'],
        scale=(320, 320),
        keep_ratio=False),
    dict(type='GenerateTrimap', kernel_size=(1, 30)),
    dict(type='PackEditInputs'),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
