_base_ = '../_base_/gen_default_runtime.py'

# config for model
dtype = 'fp16'
stable_diffusion_v15_url = 'runwayml/stable-diffusion-v1-5'

placeholder_token = '<cat-toy>'
initialize_token = 'toy'
num_vectors_per_token = 1
val_prompts = [
    'a <cat-toy> on packbag', 'a <cat-toy> on sofa',
    'a <cat-toy> in swimming pool', 'a <cat-toy>'
]

model = dict(
    type='TextualInversion',
    placeholder_token=placeholder_token,
    vae=dict(
        type='AutoencoderKL',
        from_pretrained=stable_diffusion_v15_url,
        subfolder='vae'),
    unet=dict(
        type='UNet2DConditionModel',
        from_pretrained=stable_diffusion_v15_url,
        subfolder='unet'),
    text_encoder=dict(
        type='ClipWrapper',
        clip_type='huggingface',
        pretrained_model_name_or_path=stable_diffusion_v15_url,
        subfolder='text_encoder'),
    tokenizer=stable_diffusion_v15_url,
    initialize_token=initialize_token,
    num_vectors_per_token=num_vectors_per_token,
    val_prompts=val_prompts,
    scheduler=dict(
        type='DDPMScheduler',
        from_pretrained=stable_diffusion_v15_url,
        subfolder='scheduler'),
    test_scheduler=dict(
        type='DDIMScheduler',
        from_pretrained=stable_diffusion_v15_url,
        subfolder='scheduler'),
    data_preprocessor=dict(type='DataPreprocessor', data_keys=None))

train_cfg = dict(max_iters=3000)

optim_wrapper = dict(
    modules='.*trainable_embeddings',
    optimizer=dict(type='AdamW', lr=5e-4),
    accumulative_counts=1)

pipeline = [
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='Resize', scale=(512, 512)),
    dict(type='PackInputs')
]

dataset = dict(
    type='TextualInversionDataset',
    data_root='./data/',
    concept_dir='cat_toy',
    placeholder=placeholder_token,
    template='data/imagenet_templates_small.txt',
    pipeline=pipeline)

train_dataloader = dict(
    dataset=dataset,
    num_workers=16,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    persistent_workers=True,
    batch_size=1)
val_cfg = val_evaluator = val_dataloader = None
test_cfg = test_evaluator = test_dataloader = None

default_hooks = dict(
    logger=dict(interval=10),
    checkpoint=dict(type='CheckpointHook', interval=10))
custom_hooks = [
    dict(
        type='VisualizationHook',
        interval=50,
        fixed_input=True,
        # visualize train dataset
        vis_kwargs_list=dict(type='Data', name='fake_img'),
        n_samples=1)
]
