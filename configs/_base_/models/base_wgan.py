loss_config = dict(gp_norm_mode='HWC', gp_loss_weight=10)
model = dict(
    type='WGANGP',
    data_preprocessor=dict(type='GenDataPreprocessor'),
    generator=dict(type='WGANGPGenerator', noise_size=128, out_scale=128),
    discriminator=dict(
        type='WGANGPDiscriminator',
        in_channel=3,
        in_scale=128,
        conv_module_cfg=dict(
            conv_cfg=None,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            act_cfg=dict(type='LeakyReLU', negative_slope=0.2),
            norm_cfg=dict(type='GN'),
            order=('conv', 'norm', 'act'))),
    discriminator_steps=5,
    loss_config=loss_config)
