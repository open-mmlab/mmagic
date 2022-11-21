data_preprocessor = dict(
    type='EditDataPreprocessor', mean=[127.5], std=[127.5])

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

pretrained_cfgs = dict(
    unet=dict(
        ckpt_path=
        'https://download.openmmlab.com/mmediting/synthesizers/disco/adm-u_finetuned_imagenet-512x512-ab471d70.pth',
        prefix='unet'),
    secondary_model=dict(
        ckpt_path=
        'https://download.openmmlab.com/mmediting/synthesizers/disco/secondary_model_imagenet_2.pth',
        prefix=''))

secondary_model = dict(type='SecondaryDiffusionImageNet2')

diffuser = dict(
    type='DDIMScheduler',
    variance_type='learned_range',
    beta_schedule='linear',
    clip_sample=False)

clip_models_cfg = [
    dict(type='ClipWrapper', clip_type='clip', name='ViT-B/32', jit=False),
    dict(type='ClipWrapper', clip_type='clip', name='ViT-B/16', jit=False),
    dict(type='ClipWrapper', clip_type='clip', name='RN50', jit=False)
]

# pretrained_cfgs = None
cutter_cfg = dict(
    cut_overview=eval('[12]*400+[4]*600'),
    cut_innercut=eval('[4]*400+[12]*600'),
    cut_ic_pow=eval('[1]*1000'),
    cut_icgray_p=eval('[0.2]*400+[0]*600'),
    cutn_batches=4)

loss_cfg = dict(tv_scale=0, range_scale=150, sat_scale=0, init_scale=20)

model = dict(
    type='DiscoDiffusion',
    data_preprocessor=data_preprocessor,
    unet=unet,
    diffuser=diffuser,
    secondary_model=secondary_model,
    cutter_cfg=cutter_cfg,
    loss_cfg=loss_cfg,
    clip_models_cfg=clip_models_cfg,
    use_fp16=True,
    pretrained_cfgs=pretrained_cfgs)
