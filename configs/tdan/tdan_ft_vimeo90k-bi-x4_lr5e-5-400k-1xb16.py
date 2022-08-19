_base_ = './tdan_vimeo90k-bi-x4_lr1e-4-400k-1xb16.py'

experiment_name = 'tdan_ft_vimeo90k-bi-x4_lr5e-5-400k-1xb16'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

# optimizer
optim_wrapper = dict(
    dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=5e-5),
    ))

# load_from = 'https://download.openmmlab.com/mmediting/restorers/tdan/'\
#               'tdan_vimeo90k_bix4_20210528-739979d9.pth'
