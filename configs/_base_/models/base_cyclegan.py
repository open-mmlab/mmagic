_domain_a = None  # set by user
_domain_b = None  # set by user
model = dict(
    type='CycleGAN',
    data_preprocessor=dict(type='DataPreprocessor'),
    generator=dict(
        type='ResnetGenerator',
        in_channels=3,
        out_channels=3,
        base_channels=64,
        norm_cfg=dict(type='IN'),
        use_dropout=False,
        num_blocks=9,
        padding_mode='reflect',
        init_cfg=dict(type='normal', gain=0.02)),
    discriminator=dict(
        type='PatchDiscriminator',
        in_channels=3,
        base_channels=64,
        num_conv=3,
        norm_cfg=dict(type='IN'),
        init_cfg=dict(type='normal', gain=0.02)),
    default_domain=None,  # set by user
    reachable_domains=None,  # set by user
    related_domains=None  # set by user
)
