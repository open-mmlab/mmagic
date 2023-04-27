# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch
import torch.nn as nn
from addict import Dict
from mmengine import MODELS, Config

from mmagic.utils import register_all_modules

register_all_modules()

stable_diffusion_v15_url = 'runwayml/stable-diffusion-v1-5'
unet = dict(
    type='UNet2DConditionModel',
    subfolder='unet',
    from_pretrained=stable_diffusion_v15_url)
vae = dict(
    type='AutoencoderKL',
    from_pretrained=stable_diffusion_v15_url,
    subfolder='vae')

diffusion_scheduler = dict(
    type='EditDDIMScheduler',
    variance_type='learned_range',
    beta_end=0.012,
    beta_schedule='scaled_linear',
    beta_start=0.00085,
    num_train_timesteps=1000,
    set_alpha_to_one=False,
    clip_sample=False)


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
    type='StableDiffusion',
    unet=unet,
    vae=vae,
    enable_xformers=False,
    text_encoder=dummy_text_encoder(),
    tokenizer=dummy_text_encoder(),
    scheduler=diffusion_scheduler,
    test_scheduler=diffusion_scheduler,
    tomesd_cfg=dict(ratio=0.5))


@pytest.mark.skipif(
    'win' in platform.system().lower(),
    reason='skip on windows due to limited RAM.')
def test_stable_diffusion():
    StableDiffuser = MODELS.build(Config(model))
    StableDiffuser.tokenizer = dummy_tokenizer()
    StableDiffuser.text_encoder = dummy_text_encoder()

    with pytest.raises(Exception):
        StableDiffuser.infer(1, height=64, width=64)

    with pytest.raises(Exception):
        StableDiffuser.infer('temp', height=31, width=31)

    result = StableDiffuser.infer(
        'an insect robot preparing a delicious meal',
        height=64,
        width=64,
        num_inference_steps=1,
        return_type='numpy')
    assert result['samples'].shape == (1, 3, 64, 64)

    result = StableDiffuser.infer(
        'an insect robot preparing a delicious meal',
        height=64,
        width=64,
        num_inference_steps=1,
        return_type='image')
