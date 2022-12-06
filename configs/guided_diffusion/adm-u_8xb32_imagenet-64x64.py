_base_ = [
    '../_base_/datasets/imagenet_64.py',
    '../_base_/gen_default_runtime.py',
]

model = dict(
    type='AblatedDiffusionModel',
    data_preprocessor=dict(
        type='EditDataPreprocessor', mean=[127.5], std=[127.5]),
    unet=dict(
        type='DenoisingUnet',
        image_size=64,
        in_channels=3,
        base_channels=192,
        resblocks_per_downsample=3,
        attention_res=(32, 16, 8),
        norm_cfg=dict(type='GN32', num_groups=32),
        dropout=0.1,
        num_classes=1000,
        use_fp16=False,
        resblock_updown=True,
        attention_cfg=dict(
            type='MultiHeadAttentionBlock',
            num_heads=4,
            num_head_channels=64,
            use_new_attention_order=True),
        use_scale_shift_norm=True),
    diffusion_scheduler=dict(
        type='DDPMScheduler',
        variance_type='learned_range',
        beta_schedule='squaredcos_cap_v2'),
    use_fp16=False)

test_dataloader = dict(batch_size=32, num_workers=8)

metrics = [
    dict(
        type='FrechetInceptionDistance',
        prefix='FID-Full-50k',
        fake_nums=50000,
        inception_style='StyleGAN')
]

val_evaluator = dict(metrics=metrics)
test_evaluator = dict(metrics=metrics)
