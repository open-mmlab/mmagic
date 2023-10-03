# Copyright (c) OpenMMLab. All rights reserved.
_base_ = ['../_base_/default_runtime.py']

denoiser_config = dict(
    type='KarrasDenoiser',
    sigma_data=0.5,
    sigma_max=80.0,
    sigma_min=0.002,
    weight_schedule='uniform',
)

unet_config = dict(
    type='ConsistencyUNetModel',
    in_channels=3,
    model_channels=256,
    num_res_blocks=2,
    dropout=0.0,
    channel_mult='',
    use_checkpoint=False,
    use_fp16=False,
    num_head_channels=64,
    num_heads=4,
    num_heads_upsample=-1,
    resblock_updown=True,
    use_new_attention_order=False,
    use_scale_shift_norm=False)

model = dict(
    type='ConsistencyModel',
    unet=unet_config,
    denoiser=denoiser_config,
    attention_resolutions='32,16,8',
    batch_size=4,
    class_cond=False,
    generator='determ-indiv',
    image_size=256,
    learn_sigma=False,
    model_path=
    'https://openaipublic.blob.core.windows.net/consistency/ct_cat256.pt',
    num_classes=1000,
    sampler='multistep',
    seed=42,
    training_mode='consistency_distillation',
    ts='0,62,150',
    steps=151,
    data_preprocessor=dict(
        type='DataPreprocessor', mean=[127.5] * 3, std=[127.5] * 3))
