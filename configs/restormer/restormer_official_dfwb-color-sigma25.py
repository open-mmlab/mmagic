_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/denoising-gaussian_color_test_config.py'
]

experiment_name = 'restormer_official_dfwb_color_sigma25'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

# modify sigma of RandomNoise
sigma = 25
test_dataloader = _base_.test_dataloader
for dataloader in test_dataloader:
    test_pipeline = dataloader['dataset']['pipeline']
    test_pipeline[2]['params']['gaussian_sigma'] = [sigma * 255, sigma * 255]

# model settings
model = dict(
    type='BaseEditModel',
    generator=dict(
        type='Restormer',
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='BiasFree',
        dual_pixel_task=False),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    data_preprocessor=dict(
        type='EditDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
    ))
