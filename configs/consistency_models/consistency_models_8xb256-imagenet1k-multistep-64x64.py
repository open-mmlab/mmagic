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
    model_channels=192,
    num_res_blocks=3,
    dropout=0.0,
    channel_mult='',
    use_checkpoint=False,
    use_fp16=False,
    num_head_channels=64,
    num_heads=4,
    num_heads_upsample=-1,
    resblock_updown=True,
    use_new_attention_order=False,
    use_scale_shift_norm=True)

model = dict(
    type='ConsistencyModel',
    unet=unet_config,
    denoiser=denoiser_config,
    attention_resolutions='32,16,8',
    batch_size=4,
    class_cond=True,
    generator='determ',
    image_size=64,
    learn_sigma=False,
    model_path='https://download.openxlab.org.cn/models/xiaomile/'
    'consistency_models/weight/cd_imagenet64_l2.pt',
    num_classes=1000,
    sampler='multistep',
    seed=42,
    training_mode='consistency_distillation',
    ts='0,22,39',
    data_preprocessor=dict(
        type='DataPreprocessor', mean=[127.5] * 3, std=[127.5] * 3))
