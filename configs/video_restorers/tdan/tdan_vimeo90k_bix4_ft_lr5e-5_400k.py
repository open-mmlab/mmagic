_base_ = './tdan_vimeo90k_bix4_lr1e-4_400k.py'

exp_name = 'tdan_vimeo90k_bix4_ft_lr5e-5_400k'
work_dir = f'./work_dirs/{exp_name}'

load_from = './experiments/tdan_vimeo90k_bix4_lr1e-4_400k/iter_400000.pth'

# optimizer
optim_wrapper = dict(
    dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=5e-5),
    ))
