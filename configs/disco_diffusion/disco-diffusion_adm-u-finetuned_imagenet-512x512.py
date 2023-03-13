unet = dict(
    type='DenoisingUnet',
    image_size=512,
    in_channels=3,
    base_channels=256,
    resblocks_per_downsample=2,
    attention_res=(32, 16, 8),
    norm_cfg=dict(type='GN32', num_groups=32),
    dropout=0.0,
    num_classes=0,
    use_fp16=True,
    resblock_updown=True,
    attention_cfg=dict(
        type='MultiHeadAttentionBlock',
        num_heads=4,
        num_head_channels=64,
        use_new_attention_order=False),
    use_scale_shift_norm=True)

unet_ckpt_path = 'https://download.openmmlab.com/mmediting/synthesizers/disco/adm-u_finetuned_imagenet-512x512-ab471d70.pth'  # noqa
secondary_model_ckpt_path = 'https://download.openmmlab.com/mmediting/synthesizers/disco/secondary_model_imagenet_2.pth'  # noqa
pretrained_cfgs = dict(
    unet=dict(ckpt_path=unet_ckpt_path, prefix='unet'),
    secondary_model=dict(ckpt_path=secondary_model_ckpt_path, prefix=''))

secondary_model = dict(type='SecondaryDiffusionImageNet2')

diffusion_scheduler = dict(
    type='EditDDIMScheduler',
    variance_type='learned_range',
    beta_schedule='linear',
    clip_sample=False)

clip_models = [
    dict(type='ClipWrapper', clip_type='clip', name='ViT-B/32', jit=False),
    dict(type='ClipWrapper', clip_type='clip', name='ViT-B/16', jit=False),
    dict(type='ClipWrapper', clip_type='clip', name='RN50', jit=False)
]

model = dict(
    type='DiscoDiffusion',
    unet=unet,
    diffusion_scheduler=diffusion_scheduler,
    secondary_model=secondary_model,
    clip_models=clip_models,
    use_fp16=True,
    pretrained_cfgs=pretrained_cfgs)
