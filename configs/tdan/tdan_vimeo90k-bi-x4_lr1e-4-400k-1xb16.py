_base_ = './tdan_vimeo90k-bd-x4_lr1e-4-400k-1xb16'

experiment_name = 'tdan_vimeo90k-bi-x4_lr1e-4-400k-1xb16'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

train_dataloader = dict(dataset=dict(data_prefix=dict(img='BIx4', gt='GT')))

val_dataloader = dict(dataset=dict(data_prefix=dict(img='BIx4', gt='GT')))

test_dataloader = dict(dataset=dict(data_prefix=dict(img='BIx4', gt='GT')))

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=400_000, val_interval=50000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# No learning policy
