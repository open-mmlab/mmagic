_base_ = ['swinir_x3s48w8d6e180_8xb4-lr2e-4-500k_div2k.py']

experiment_name = 'swinir_x3s64w8d4e60_8xb4-lr2e-4-500k_div2k'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

scale = 3
img_size = 64

# model settings
model = _base_.model
model['generator']['img_size'] = img_size
model['generator']['depths'] = [6, 6, 6, 6]
model['generator']['embed_dim'] = 60
model['generator']['num_heads'] = [6, 6, 6, 6]
model['generator']['upsampler'] = 'pixelshuffledirect'

# modify patch size of train_dataloader
train_dataloader = _base_.train_dataloader
train_pipeline = train_dataloader['dataset']['pipeline']
train_pipeline[3]['gt_patch_size'] = img_size * scale