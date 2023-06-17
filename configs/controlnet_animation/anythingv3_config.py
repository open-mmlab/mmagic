# config for model
stable_diffusion_v15_url = 'Linaqruf/anything-v3.0'
controlnet_hed_url = 'lllyasviel/sd-controlnet-hed'
control_detector = 'lllyasviel/ControlNet'
control_scheduler = 'UniPCMultistepScheduler'

# method type : 'multi-frame rendering' or 'attention_injection'
inference_method = 'attention_injection'

model = dict(
    type='ControlStableDiffusionImg2Img',
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
        type='ControlNetModel', from_pretrained=controlnet_hed_url),
    scheduler=dict(
        type='DDPMScheduler',
        from_pretrained=stable_diffusion_v15_url,
        subfolder='scheduler'),
    test_scheduler=dict(
        type='DDIMScheduler',
        from_pretrained=stable_diffusion_v15_url,
        subfolder='scheduler'),
    data_preprocessor=dict(type='DataPreprocessor'),
    init_cfg=dict(type='init_from_unet'),
    enable_xformers=False,
)
