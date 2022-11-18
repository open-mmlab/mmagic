_base_ = [
    '../_base_/gen_default_runtime.py',
    '../_base_/models/sagan/base_sagan_128x128.py',
    '../_base_/datasets/imagenet_128.py',
]

# MODEL
init_cfg = dict(type='studio')
model = dict(
    num_classes=1000,
    generator=dict(num_classes=1000, init_cfg=init_cfg),
    discriminator=dict(num_classes=1000, init_cfg=init_cfg))

# TRAIN
train_cfg = dict(
    max_iters=1000000, val_interval=10000, dynamic_intervals=[(800000, 4000)])
train_dataloader = dict(batch_size=64)  # train on 4 gpus

optim_wrapper = dict(
    generator=dict(optimizer=dict(type='Adam', lr=0.0001, betas=(0.0, 0.999))),
    discriminator=dict(
        optimizer=dict(type='Adam', lr=0.0004, betas=(0.0, 0.999))))

# VIS_HOOK
custom_hooks = [
    dict(
        type='GenVisualizationHook',
        interval=5000,
        fixed_input=True,
        vis_kwargs_list=dict(type='GAN', name='fake_img'))
]

# METRICS
inception_pkl = './work_dirs/inception_pkl/imagenet-full.pkl'
metrics = [
    dict(
        type='InceptionScore',
        prefix='IS-50k',
        fake_nums=50000,
        inception_style='StyleGAN',
        sample_model='orig'),
    dict(
        type='FrechetInceptionDistance',
        prefix='FID-Full-50k',
        fake_nums=50000,
        inception_style='StyleGAN',
        inception_pkl=inception_pkl,
        sample_model='orig')
]
# save multi best checkpoints
default_hooks = dict(
    checkpoint=dict(
        save_best=['FID-Full-50k/fid', 'IS-50k/is'], rule=['less', 'greater']))

val_dataloader = test_dataloader = dict(batch_size=64)
val_evaluator = test_evaluator = dict(metrics=metrics)
