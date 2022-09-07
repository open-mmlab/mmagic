model = dict(
    type='SinGAN',
    data_preprocessor=dict(
        type='GenDataPreprocessor', non_image_keys=['input_sample']),
    generator=dict(
        type='SinGANMultiScaleGenerator',
        in_channels=3,
        out_channels=3,
        num_scales=None,  # need to be specified
    ),
    discriminator=dict(
        type='SinGANMultiScaleDiscriminator',
        in_channels=3,
        num_scales=None,  # need to be specified
    ),
    noise_weight_init=0.1,
    iters_per_scale=None,
    test_pkl_data=None,
    lr_scheduler_args=dict(milestones=[1600], gamma=0.1))
