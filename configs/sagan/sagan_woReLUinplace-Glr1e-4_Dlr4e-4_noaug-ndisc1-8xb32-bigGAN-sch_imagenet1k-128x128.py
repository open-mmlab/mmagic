# In this config, we follow the setting `launch_SAGAN_bz128x2_ema.sh` from
# BigGAN's repo. Please refer to https://github.com/ajbrock/BigGAN-PyTorch/blob/master/scripts/launch_SAGAN_bs128x2_ema.sh  # noqa
# In summary, in this config:
# 1. use eps=1e-8 for Spectral Norm
# 2. not use syncBN
# 3. not use  Spectral Norm for embedding layers in cBN
# 4. start EMA at iterations
# 5. use xavier_uniform for weight initialization
# 6. no data augmentation

_base_ = [
    '../_base_/gen_default_runtime.py',
    '../_base_/models/sagan/base_sagan_128x128.py',
    '../_base_/datasets/imagenet_noaug_128.py',
]

# MODEL
init_cfg = dict(type='BigGAN')
model = dict(
    num_classes=1000,
    generator=dict(
        num_classes=1000,
        init_cfg=init_cfg,
        norm_eps=1e-5,
        sn_eps=1e-8,
        auto_sync_bn=False,
        with_embedding_spectral_norm=False),
    discriminator=dict(num_classes=1000, init_cfg=init_cfg, sn_eps=1e-8),
    discriminator_steps=1,
    ema_config=dict(interval=1, momentum=0.999, start_iter=2000))

# TRAINING
train_cfg = dict(
    max_iters=1000000, val_interval=10000, dynamic_intervals=[(800000, 4000)])
train_dataloader = dict(batch_size=32)  # train on 8 gpus

optim_wrapper = dict(
    generator=dict(optimizer=dict(type='Adam', lr=0.0001, betas=(0.0, 0.999))),
    discriminator=dict(
        optimizer=dict(type='Adam', lr=0.0004, betas=(0.0, 0.999))))

# VIS_HOOK
custom_hooks = [
    dict(
        type='VisualizationHook',
        interval=5000,
        fixed_input=True,
        # vis ema and orig at the same time
        vis_kwargs_list=dict(
            type='Noise',
            name='fake_img',
            sample_model='ema/orig',
            target_keys=['ema.fake_img', 'orig.fake_img']))
]

# METRICS
inception_pkl = './work_dirs/inception_pkl/imagenet-full.pkl'
metrics = [
    dict(
        type='InceptionScore',
        prefix='IS-50k',
        fake_nums=50000,
        inception_style='StyleGAN',
        sample_model='ema'),
    dict(
        type='FrechetInceptionDistance',
        prefix='FID-Full-50k',
        fake_nums=50000,
        inception_style='StyleGAN',
        inception_pkl=inception_pkl,
        sample_model='ema')
]
# save multi best checkpoints
default_hooks = dict(
    checkpoint=dict(
        save_best=['FID-Full-50k/fid', 'IS-50k/is'], rule=['less', 'greater']))

val_dataloader = test_dataloader = dict(batch_size=64)
val_evaluator = test_evaluator = dict(metrics=metrics)
