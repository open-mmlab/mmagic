_base_ = ['swinir_x3s48w8d6e180_8xb4-lr2e-4-500k_div2k.py']

experiment_name = 'swinir_x3s64w8d6e180_8xb4-lr2e-4-500k_df2k'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

scale = 3
img_size = 64

# model settings
model = dict(generator=dict(img_size=img_size))

# modify patch size of train_pipeline
train_pipeline = _base_.train_pipeline
train_pipeline[3]['gt_patch_size'] = img_size * scale

# dataset settings
dataset_type = 'BasicImageDataset'
data_root = 'data'

train_dataloader = dict(
    num_workers=4,
    batch_size=4,
    drop_last=True,
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file='meta_info_DF2K3450sub_GT.txt',
        metainfo=dict(dataset_type='div2k', task_name='sisr'),
        data_root=data_root + '/DF2K',
        data_prefix=dict(
            img='DF2K_train_LR_bicubic/X3_sub', gt='DF2K_train_HR_sub'),
        filename_tmpl=dict(img='{}', gt='{}'),
        pipeline=train_pipeline))
