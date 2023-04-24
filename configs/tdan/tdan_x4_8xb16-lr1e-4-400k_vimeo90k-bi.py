_base_ = './tdan_x4_8xb16-lr1e-4-400k_vimeo90k-bd.py'

experiment_name = 'tdan_x4_1xb16-lr1e-4-400k_vimeo90k-bi'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

train_dataloader = dict(dataset=dict(data_prefix=dict(img='BIx4', gt='GT')))

val_dataloader = dict(dataset=dict(data_prefix=dict(img='BIx4', gt='GT')))

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=400_000, val_interval=50000)
val_cfg = dict(type='MultiValLoop')

# No learning policy
