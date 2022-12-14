_base_ = ['swinir_s128w8d6e180_8xb1-lr2e-4-1600k_colorDN15_dfwb.py']

experiment_name = 'swinir_s128w8d6e180_8xb1-lr2e-4-1600k_colorDN25_dfwb'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

# modify sigma of RandomNoise
sigma = 25
test_dataloader = _base_.test_dataloader
for dataloader in test_dataloader:
    test_pipeline = dataloader['dataset']['pipeline']
    test_pipeline[2]['params']['gaussian_sigma'] = [sigma * 255, sigma * 255]

train_dataloader = _base_.train_dataloader
train_pipeline = train_dataloader['dataset']['pipeline']
train_pipeline[-2]['params']['gaussian_sigma'] = [sigma * 255, sigma * 255]

val_dataloader = _base_.val_dataloader
val_pipeline = val_dataloader['dataset']['pipeline']
val_pipeline[2]['params']['gaussian_sigma'] = [sigma * 255, sigma * 255]
