_base_ = '../_base_/gen_default_runtime.py'

# config for model
stable_diffusion_v15_url = 'runwayml/stable-diffusion-v1-5'
controlnet_canny_url = 'lllyasviel/sd-controlnet-canny'

model = dict(
    type='ControlStableDiffusion',
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
    controlnet=dict(
        type='ControlNetModel',
        # from_pretrained=controlnet_canny_rul
        from_config=controlnet_canny_url  # train from scratch
    ),
    scheduler=dict(
        type='DDPMScheduler',
        from_pretrained=stable_diffusion_v15_url,
        subfolder='scheduler'),
    test_scheduler=dict(
        type='DDIMScheduler',
        from_pretrained=stable_diffusion_v15_url,
        subfolder='scheduler'),
    data_preprocessor=dict(type='DataPreprocessor'),
    init_cfg=dict(type='init_from_unet'))

# config for training
train_cfg = dict(max_iters=10000)
optim_wrapper = dict(controlnet=dict(optimizer=dict(type='AdamW', lr=1e-5)))

# Config for data loader
pipeline = [
    dict(type='LoadImageFromFile', key='source', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='target', channel_order='rgb'),
    dict(
        type='PackInputs',
        keys=['source', 'target'],
        data_keys='prompt',
        meta_keys=[
            'source_channel_order', 'source_color_type',
            'target_channel_order', 'target_color_type'
        ])
]
dataset = dict(
    type='ControlNetDataset',
    data_root='./data/fill50k',
    ann_file='prompt.json',
    pipeline=pipeline)
train_dataloader = dict(
    dataset=dataset,
    num_workers=16,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    persistent_workers=True,
    batch_size=4)
val_cfg = val_evaluator = val_dataloader = None
test_cfg = test_evaluator = test_dataloader = None

# hooks
custom_hooks = [
    dict(
        type='VisualizationHook',
        interval=300,
        fixed_input=True,
        # visualize train dataset
        vis_kwargs_list=dict(type='Data', name='fake_img'),
        n_samples=4,
        n_row=2)
]
