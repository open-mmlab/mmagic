# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch
import torch.nn as nn
from addict import Dict
from mmengine import MODELS, Config

from mmagic.utils import register_all_modules

register_all_modules()

unet = dict(
    type='DenoisingUnet',
    image_size=128,
    base_channels=32,
    channels_cfg=[1, 2],
    unet_type='stable',
    act_cfg=dict(type='silu', inplace=False),
    cross_attention_dim=768,
    num_heads=2,
    in_channels=9,
    out_channels=4,
    layers_per_block=1,
    down_block_types=['CrossAttnDownBlock2D', 'DownBlock2D'],
    up_block_types=['UpBlock2D', 'CrossAttnUpBlock2D'],
    output_cfg=dict(var='fixed'))

vae = dict(
    type='EditAutoencoderKL',
    act_fn='silu',
    block_out_channels=[128],
    down_block_types=['DownEncoderBlock2D'],
    in_channels=3,
    latent_channels=4,
    layers_per_block=1,
    norm_num_groups=32,
    out_channels=3,
    sample_size=128,
    up_block_types=[
        'UpDecoderBlock2D',
    ])

diffusion_scheduler = dict(
    type='EditDDIMScheduler',
    variance_type='learned_range',
    beta_end=0.012,
    beta_schedule='scaled_linear',
    beta_start=0.00085,
    num_train_timesteps=1000,
    set_alpha_to_one=False,
    clip_sample=False)

init_cfg = dict(type='Pretrained', pretrained_model_path=None)


class dummy_tokenizer(nn.Module):

    def __init__(self):
        super().__init__()
        self.model_max_length = 0

    def __call__(self,
                 prompt,
                 padding='max_length',
                 max_length=0,
                 truncation=False,
                 return_tensors='pt'):
        text_inputs = Dict()
        text_inputs['input_ids'] = torch.ones([1, 77])
        text_inputs['attention_mask'] = torch.ones([1, 77])
        return text_inputs


class dummy_text_encoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.config = None

    def __call__(self, x, attention_mask):
        result = torch.rand([1, 77, 768])
        return [result]


model = dict(
    type='StableDiffusionInpaint',
    scheduler=diffusion_scheduler,
    unet=unet,
    vae=vae,
    init_cfg=init_cfg,
    text_encoder=dummy_text_encoder(),
    tokenizer=dummy_text_encoder())


@pytest.mark.skipif(
    'win' in platform.system().lower(),
    reason='skip on windows due to limited RAM.')
def test_stable_diffusion():
    StableDiffuser = MODELS.build(Config(model))
    StableDiffuser.tokenizer = dummy_tokenizer()
    StableDiffuser.text_encoder = dummy_text_encoder()
    config = getattr(StableDiffuser.vae, 'config', None)
    if config is None:

        class DummyConfig:
            pass

        config = DummyConfig()
        setattr(config, 'scaling_factor', 1.2)
        setattr(StableDiffuser.vae, 'config', config)

    image = torch.clip(torch.randn((1, 3, 64, 64)), -1, 1)
    mask = torch.clip(torch.randn((1, 1, 64, 64)), 0, 1)

    with pytest.raises(Exception):
        StableDiffuser.infer('temp', image, mask, height=31, width=31)

    result = StableDiffuser.infer(
        'an insect robot preparing a delicious meal',
        image=image,
        mask_image=mask,
        height=64,
        width=64,
        num_inference_steps=1,
        return_type='numpy')
    assert result['samples'].shape == (1, 3, 64, 64)

    result = StableDiffuser.infer(
        'an insect robot preparing a delicious meal',
        image=image,
        mask_image=mask,
        height=64,
        width=64,
        num_inference_steps=1,
        return_type='image')


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
