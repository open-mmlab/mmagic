# Copyright (c) OpenMMLab. All rights reserved.
model = dict(
    type='BigGAN',
    num_classes=1000,
    data_preprocessor=dict(type='DataPreprocessor'),
    generator=dict(
        type='BigGANGenerator',
        output_scale=128,
        noise_size=120,
        num_classes=1000,
        base_channels=96,
        shared_dim=128,
        with_shared_embedding=True,
        sn_eps=1e-6,
        act_cfg=dict(type='ReLU', inplace=True),
        split_noise=True,
        auto_sync_bn=False,
        init_cfg=dict(type='ortho')),
    discriminator=dict(
        type='BigGANDiscriminator',
        input_scale=128,
        num_classes=1000,
        base_channels=96,
        sn_eps=1e-6,
        act_cfg=dict(type='ReLU', inplace=True),
        with_spectral_norm=True,
        init_cfg=dict(type='ortho')),
    generator_steps=1,
    discriminator_steps=1)
