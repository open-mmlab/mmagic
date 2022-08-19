_base_ = './tdan_x4_lr1e-4-400k-1xb16_vimeo90k-bd.py'

experiment_name = 'tdan_x4ft_lr5e-5-800k-1xb16_vimeo90k-bd'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

# optimizer
optim_wrapper = dict(
    dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=5e-5),
    ))

# load_from = 'https://download.openmmlab.com/mmediting/restorers/tdan/'\
#               'tdan_vimeo90k_bdx4_20210528-c53ab844.pth'
