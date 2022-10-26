_base_ = [
    '../_base_/models/sngan_proj/base_sngan_proj_128x128.py',
    '../_base_/datasets/imagenet_128.py',
    '../_base_/gen_default_runtime.py',
]

# MODEL
discriminator_steps = 5
num_classes = 1000
init_cfg = dict(type='studio')
model = dict(
    num_classes=num_classes,
    generator=dict(
        num_classes=num_classes,
        act_cfg=dict(type='ReLU', inplace=True),
        init_cfg=init_cfg),
    discriminator=dict(
        num_classes=num_classes,
        act_cfg=dict(type='ReLU', inplace=True),
        init_cfg=init_cfg),
    discriminator_steps=discriminator_steps)

# TRAINING
train_cfg = dict(
    max_iters=500000 * discriminator_steps,
    val_interval=10000,
    dynamic_intervals=[(800000, 4000)])
train_dataloader = dict(batch_size=128)  # train on 2 gpus

optim_wrapper = dict(
    generator=dict(optimizer=dict(type='Adam', lr=0.0002, betas=(0.5, 0.999))),
    discriminator=dict(
        optimizer=dict(type='Adam', lr=0.0002, betas=(0.5, 0.999))))

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

# EVALUATION
val_dataloader = test_dataloader = dict(batch_size=128)
val_evaluator = test_evaluator = dict(metrics=metrics)
