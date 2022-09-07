_base_ = [
    '../_base_/gen_default_runtime.py',
    '../_base_/datasets/grow_scale_imgs_128x128.py',
]

# define GAN model
model = dict(
    type='ProgressiveGrowingGAN',
    data_preprocessor=dict(type='GenDataPreprocessor'),
    noise_size=512,
    generator=dict(type='PGGANGenerator', out_scale=128),
    discriminator=dict(type='PGGANDiscriminator', in_scale=128),
    nkimgs_per_scale={
        '4': 600,
        '8': 1200,
        '16': 1200,
        '32': 1200,
        '64': 1200,
        '128': 12000
    },
    transition_kimgs=600,
    ema_config=dict(interval=1))

# MODEL
model_wrapper_cfg = dict(find_unused_parameters=True)

# TRAIN
train_cfg = dict(max_iters=280000)
data_roots = {'128': './data/celeba-cropped/cropped_images_aligned_png'}
train_dataloader = dict(batch_size=64, dataset=dict(data_roots=data_roots))

optim_wrapper = dict(
    constructor='PGGANOptimWrapperConstructor',
    generator=dict(optimizer=dict(type='Adam', lr=0.001, betas=(0., 0.99))),
    discriminator=dict(
        optimizer=dict(type='Adam', lr=0.001, betas=(0., 0.99))),
    lr_schedule=dict(generator={'128': 0.0015}, discriminator={'128': 0.0015}))

# VIS_HOOK + DATAFETCH
custom_hooks = [
    dict(
        type='GenVisualizationHook',
        interval=5000,
        fixed_input=True,
        # vis ema and orig at the same time
        vis_kwargs_list=dict(
            type='Noise',
            name='fake_img',
            sample_model='ema/orig',
            target_keys=['ema', 'orig'])),
    dict(type='PGGANFetchDataHook')
]

# METRICS
metrics = [
    dict(type='SWD', fake_nums=16384, image_shape=(3, 128, 128), prefix='SWD'),
    dict(type='MS_SSIM', fake_nums=10000, prefix='MS-SSIM')
]

# do not evaluate in training
val_cfg = val_evaluator = val_dataloader = None

test_evaluator = dict(metrics=metrics)
test_dataloader = dict(dataset=dict(data_root=data_roots['128']))
