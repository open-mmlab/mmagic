_base_ = '../_base_/gen_default_runtime.py'

randomness = dict(seed=2023, diff_rank_seed=True)
# dtype="fp32"
# config for model
stable_diffusion_v15_url = 'runwayml/stable-diffusion-v1-5'

data_root = './data/vico'
concept_dir = 'dog7'

# 1 for using image cross
image_cross_layers = [
    # down blocks (2x transformer block) * (3x down blocks) = 6
    0,
    0,
    0,
    0,
    0,
    0,
    # mid block (1x transformer block) * (1x mid block)= 1
    0,
    # up blocks (3x transformer block) * (3x up blocks) = 9
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
]
reg_loss_weight: float = 5e-4
placeholder: str = 'S*'
val_prompts = ['a photo of a S*']
initialize_token: str = 'dog'
num_vectors_per_token: int = 1

model = dict(
    type='ViCo',
    vae=dict(
        type='AutoencoderKL',
        from_pretrained=stable_diffusion_v15_url,
        subfolder='vae'),
    unet=dict(
        type='UNet2DConditionModel',
        subfolder='unet',
        from_pretrained=stable_diffusion_v15_url),
    text_encoder=dict(
        type='ClipWrapper',
        clip_type='huggingface',
        pretrained_model_name_or_path=stable_diffusion_v15_url,
        subfolder='text_encoder'),
    tokenizer=stable_diffusion_v15_url,
    scheduler=dict(
        type='DDPMScheduler',
        from_pretrained=stable_diffusion_v15_url,
        subfolder='scheduler'),
    test_scheduler=dict(
        type='DDIMScheduler',
        from_pretrained=stable_diffusion_v15_url,
        subfolder='scheduler'),
    # dtype=dtype,
    data_preprocessor=dict(type='DataPreprocessor', data_keys=None),
    image_cross_layers=image_cross_layers,
    reg_loss_weight=reg_loss_weight,
    placeholder=placeholder,
    initialize_token=initialize_token,
    num_vectors_per_token=num_vectors_per_token,
    val_prompts=val_prompts)

train_cfg = dict(max_iters=500)

paramwise_cfg = dict(
    custom_keys={
        'image_cross_attention': dict(lr_mult=2e-3),
        'trainable_embeddings': dict(lr_mult=1.0)
    })
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.005, weight_decay=0.01),
    constructor='DefaultOptimWrapperConstructor',
    paramwise_cfg=paramwise_cfg,
    accumulative_counts=1)

pipeline = [
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='img_ref', channel_order='rgb'),
    dict(type='Resize', keys=['img', 'img_ref'], scale=(512, 512)),
    dict(
        type='PackInputs',
        keys=['img', 'img_ref'],
        data_keys='prompt',
        meta_keys=[
            'img_channel_order', 'img_color_type', 'img_ref_channel_order',
            'img_ref_color_type'
        ])
]
dataset = dict(
    type='TextualInversionDataset',
    data_root=data_root,
    concept_dir=concept_dir,
    placeholder=placeholder,
    template='data/vico/imagenet_templates_small.txt',
    with_image_reference=True,
    pipeline=pipeline)
train_dataloader = dict(
    dataset=dataset,
    num_workers=16,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    persistent_workers=True,
    batch_size=1)
val_cfg = val_evaluator = val_dataloader = None
test_cfg = test_evaluator = test_dataloader = None

# hooks
default_hooks = dict(logger=dict(interval=10))
custom_hooks = [
    dict(
        type='VisualizationHook',
        interval=50,
        fixed_input=True,
        # visualize train dataset
        vis_kwargs_list=dict(type='Data', name='fake_img'),
        n_samples=1)
]
