_base_ = [
    'restormer_official_dpdd-single.py',
]

experiment_name = 'restormer_official_dpdd-dual'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

# modify PackEditInputs
test_dataloader = _base_.test_dataloader
for dataloader in test_dataloader:
    test_pipeline = dataloader['dataset']['pipeline']
    test_pipeline[4] = dict(type='PackEditInputs', keys=['imgL', 'imgR'])

# model settings
model = dict(
    generator=dict(inp_channels=6, dual_pixel_task=True),
    data_preprocessor=dict(type='GenDataPreprocessor'))
