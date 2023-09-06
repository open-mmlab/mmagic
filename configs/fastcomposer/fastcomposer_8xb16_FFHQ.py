_base_ = '../_base_/gen_default_runtime.py'

# config for model
stable_diffusion_v15_url = 'runwayml/stable-diffusion-v1-5'
clip_vit_url = 'openai/clip-vit-large-patch14'
finetuned_model_path = 'https://download.openxlab.org.cn/models/xiaomile/'\
                       'fastcomposer/weight/pytorch_model.bin'

model = dict(
    type='FastComposer',
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
    pretrained_cfg=dict(
        finetuned_model_path=finetuned_model_path,
        enable_xformers_memory_efficient_attention=None,
        pretrained_model_name_or_path=stable_diffusion_v15_url,
        image_encoder=clip_vit_url,
        revision=None,
        non_ema_revision=None,
        object_localization=None,
        object_localization_weight=0.01,
        localization_layers=5,
        mask_loss=None,
        mask_loss_prob=0.5,
        object_localization_threshold=1.0,
        object_localization_normalize=None,
        no_object_augmentation=True,
        object_resolution=256),
    scheduler=dict(
        type='DDPMScheduler',
        from_pretrained=stable_diffusion_v15_url,
        subfolder='scheduler'),
    test_scheduler=dict(
        type='DDIMScheduler',
        from_pretrained=stable_diffusion_v15_url,
        subfolder='scheduler'),
    dtype='fp32',
    data_preprocessor=dict(type='DataPreprocessor'))
