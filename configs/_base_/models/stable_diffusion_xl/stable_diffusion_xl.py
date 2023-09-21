# Use DiffuserWrapper!
stable_diffusion_xl_url = 'stabilityai/stable-diffusion-xl-base-1.0'
vae_url = 'madebyollin/sdxl-vae-fp16-fix'
unet = dict(
    type='UNet2DConditionModel',
    subfolder='unet',
    from_pretrained=stable_diffusion_xl_url)
vae = dict(type='AutoencoderKL', from_pretrained=vae_url)

diffusion_scheduler = dict(
    type='DDPMScheduler',
    from_pretrained=stable_diffusion_xl_url,
    subfolder='scheduler')

model = dict(
    type='StableDiffusionXL',
    dtype='fp16',
    with_cp=True,
    unet=unet,
    vae=vae,
    enable_xformers=False,
    text_encoder_one=dict(
        type='ClipWrapper',
        clip_type='huggingface',
        pretrained_model_name_or_path=stable_diffusion_xl_url,
        subfolder='text_encoder'),
    tokenizer_one=stable_diffusion_xl_url,
    text_encoder_two=dict(
        type='ClipWrapper',
        clip_type='huggingface',
        pretrained_model_name_or_path=stable_diffusion_xl_url,
        subfolder='text_encoder_2'),
    tokenizer_two=stable_diffusion_xl_url,
    scheduler=diffusion_scheduler,
    test_scheduler=diffusion_scheduler,
    data_preprocessor=dict(type='DataPreprocessor', data_keys=None))
