_base_ = './tdan_x4_1xb16-lr1e-4-400k_vimeo90k-bi.py'

experiment_name = 'tdan_x4ft_1xb16-lr5e-5-400k_vimeo90k-bi'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

# optimizer
optim_wrapper = dict(
    constructor='DefaultOptimWrapperConstructor',
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=5e-5),
)

# load_from = 'https://download.openmmlab.com/mmediting/restorers/tdan/'\
#               'tdan_vimeo90k_bix4_20210528-739979d9.pth'
