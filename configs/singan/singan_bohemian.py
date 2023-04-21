_base_ = ['./singan_fish.py']

# MODEL
# NOTE: add by user, e.g.:
# test_pkl_data = './work_dirs/singan_pkl/singan_bohemian_20210406_175439-f964ee38.pkl'  # noqa
test_pkl_data = None
model = dict(test_pkl_data=test_pkl_data)

# HOOKS
custom_hooks = [
    dict(
        type='PickleDataHook',
        output_dir='pickle',
        interval=-1,
        after_run=True,
        data_name_list=['noise_weights', 'fixed_noises', 'curr_stage']),
    dict(
        type='VisualizationHook',
        interval=5000,
        fixed_input=True,
        vis_kwargs_list=dict(type='SinGAN', name='bohemian'))
]

# DATA
min_size = 25
max_size = 500
data_root = './data/singan/bohemian.png'
train_dataloader = dict(
    dataset=dict(data_root=data_root, min_size=min_size, max_size=max_size))
