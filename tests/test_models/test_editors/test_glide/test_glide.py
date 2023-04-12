# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from mmengine import MODELS, Config

from mmedit.utils import register_all_modules

register_all_modules()

unet = dict(
    type='Text2ImUNet',
    image_size=64,
    base_channels=192,
    in_channels=3,
    resblocks_per_downsample=3,
    attention_res=(32, 16, 8),
    norm_cfg=dict(type='GN32', num_groups=32),
    dropout=0.1,
    num_classes=0,
    use_fp16=False,
    resblock_updown=True,
    attention_cfg=dict(
        type='MultiHeadAttentionBlock',
        num_heads=1,
        num_head_channels=64,
        use_new_attention_order=False,
        encoder_channels=512),
    use_scale_shift_norm=True,
    text_ctx=128,
    xf_width=512,
    xf_layers=16,
    xf_heads=8,
    xf_final_ln=True,
    xf_padding=True,
)

diffusion_scheduler = dict(
    type='EditDDIMScheduler',
    variance_type='learned_range',
    beta_schedule='squaredcos_cap_v2')

unet_up = dict(
    type='SuperResText2ImUNet',
    image_size=256,
    base_channels=192,
    in_channels=3,
    output_cfg=dict(var='FIXED'),
    resblocks_per_downsample=2,
    attention_res=(32, 16, 8),
    norm_cfg=dict(type='GN32', num_groups=32),
    dropout=0.1,
    num_classes=0,
    use_fp16=False,
    resblock_updown=True,
    attention_cfg=dict(
        type='MultiHeadAttentionBlock',
        num_heads=1,
        num_head_channels=64,
        use_new_attention_order=False,
        encoder_channels=512),
    use_scale_shift_norm=True,
    text_ctx=128,
    xf_width=512,
    xf_layers=16,
    xf_heads=8,
    xf_final_ln=True,
    xf_padding=True,
)

diffusion_scheduler_up = dict(
    type='EditDDIMScheduler',
    variance_type='learned_range',
    beta_schedule='linear')

model = dict(
    type='Glide',
    data_preprocessor=dict(
        type='EditDataPreprocessor', mean=[127.5], std=[127.5]),
    unet=unet,
    diffusion_scheduler=diffusion_scheduler,
    unet_up=unet_up,
    diffusion_scheduler_up=diffusion_scheduler_up,
    use_fp16=False)


def test_glide():
    glide = MODELS.build(Config(model))
    prompt = 'an oil painting of a corgi'

    with pytest.raises(Exception):
        glide.infer(
            prompt=prompt,
            batch_size=1,
            num_inference_steps=1,
            num_inference_steps_up=1)

    result = glide.infer(
        init_image=torch.randn(1, 3, 64, 64),
        prompt=prompt,
        batch_size=1,
        guidance_scale=3.0,
        num_inference_steps=1,
        num_inference_steps_up=1)
    assert result['samples'].shape == (1, 3, 256, 256)
