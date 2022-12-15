_base_ = ['swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR10.py']

experiment_name = 'swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR40'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

# modify JPEG quality factor of RandomJPEGCompression
quality = 40
train_dataloader = _base_.train_dataloader
train_pipeline = train_dataloader['dataset']['pipeline']
train_pipeline[-2]['params']['quality'] = [quality, quality]

val_dataloader = _base_.val_dataloader
val_pipeline = val_dataloader['dataset']['pipeline']
val_pipeline[2]['params']['quality'] = [quality, quality]

test_dataloader = _base_.test_dataloader
for dataloader in test_dataloader:
    test_pipeline = dataloader['dataset']['pipeline']
    test_pipeline[2]['params']['quality'] = [quality, quality]
