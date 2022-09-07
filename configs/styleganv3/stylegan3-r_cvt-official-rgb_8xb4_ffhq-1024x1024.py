_base_ = [
    '../_base_/models/base_styleganv3.py',
    '../_base_/datasets/ffhq_flip.py',
    '../_base_/gen_default_runtime.py',
]

synthesis_cfg = {
    'type': 'SynthesisNetwork',
    'channel_base': 65536,
    'channel_max': 1024,
    'magnitude_ema_beta': 0.999,
    'conv_kernel': 1,
    'use_radial_filters': True
}

r1_gamma = 32.8
d_reg_interval = 16

model = dict(
    generator=dict(
        out_size=1024,
        img_channels=3,
        synthesis_cfg=synthesis_cfg,
        rgb2bgr=True),
    discriminator=dict(type='StyleGAN2Discriminator', in_size=1024))

batch_size = 4
data_root = './data/ffhq/images'

train_dataloader = dict(
    batch_size=batch_size, dataset=dict(data_root=data_root))

val_dataloader = dict(batch_size=batch_size, dataset=dict(data_root=data_root))

test_dataloader = dict(
    batch_size=batch_size, dataset=dict(data_root=data_root))

train_cfg = train_dataloader = optim_wrapper = None

metrics = [
    dict(
        type='FrechetInceptionDistance',
        prefix='FID-Full-50k',
        fake_nums=50000,
        inception_style='StyleGAN',
        sample_model='ema')
]
# NOTE: config for save multi best checkpoints
# default_hooks = dict(
#     checkpoint=dict(
#         save_best=['FID-Full-50k/fid', 'IS-50k/is'],
#         rule=['less', 'greater']))
default_hooks = dict(checkpoint=dict(save_best='FID-Full-50k/fid'))
val_evaluator = dict(metrics=metrics)
test_evaluator = dict(metrics=metrics)
