_base_ = './tdan_x4_8xb16-lr1e-4-400k_vimeo90k-bd.py'

experiment_name = 'tdan_x4ft_1xb16-lr5e-5-800k_vimeo90k-bd'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

# optimizer
optim_wrapper = dict(
    constructor='DefaultOptimWrapperConstructor',
    type='OptimWrapper',
    optimizer=dict(_delete_=True, type='Adam', lr=5e-5),
)

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=800_000, val_interval=50000)

# load_from = 'tdan_x4_1xb16-lr1e-4-400k_vimeo90k-bd/iter_400000.pth'
