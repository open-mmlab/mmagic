# Copyright (c) OpenMMLab. All rights reserved.
import torch
from addict import Dict
from mmengine import MODELS, Config

from mmedit.utils import register_all_modules

register_all_modules()

unet = dict(
    type='DenoisingUnet',
    image_size=512,
    base_channels=320,
    channels_cfg=[1, 2, 4, 4],
    unet_type='stable',
    act_cfg=dict(type='silu', inplace=False),
    cross_attention_dim=768,
    num_heads=8,
    in_channels=4,
    layers_per_block=2,
    down_block_types=[
        'CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'CrossAttnDownBlock2D',
        'DownBlock2D'
    ],
    up_block_types=[
        'UpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D',
        'CrossAttnUpBlock2D'
    ],
    output_cfg=dict(var='fixed'))

vae = dict(
    act_fn='silu',
    block_out_channels=[128, 256, 512, 512],
    down_block_types=[
        'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D',
        'DownEncoderBlock2D'
    ],
    in_channels=3,
    latent_channels=4,
    layers_per_block=2,
    norm_num_groups=32,
    out_channels=3,
    sample_size=512,
    up_block_types=[
        'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D',
        'UpDecoderBlock2D'
    ])

diffusion_scheduler = dict(
    type='DDIMScheduler',
    variance_type='learned_range',
    beta_end=0.012,
    beta_schedule='scaled_linear',
    beta_start=0.00085,
    num_train_timesteps=1000,
    set_alpha_to_one=False,
    clip_sample=False)

pretrained_ckpt_path = dict(
    tokenizer=None,
    text_encoder=None,
    feature_extractor=None,
    safety_checker=None)

model = dict(
    type='StableDiffusion',
    diffusion_scheduler=diffusion_scheduler,
    unet=unet,
    vae=vae,
    pretrained_ckpt_path=pretrained_ckpt_path,
    requires_safety_checker=False,
)


class dummy_tokenizer:

    def __init__(self):
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


class dummy_text_encoder:

    def __init__(self):
        self.config = None

    def __call__(self, x, attention_mask):
        result = torch.rand([1, 77, 768])
        return [result]


def test_stable_diffusion():
    StableDiffuser = MODELS.build(Config(model))
    StableDiffuser.tokenizer = dummy_tokenizer()
    StableDiffuser.text_encoder = dummy_text_encoder()
    result = StableDiffuser.infer(
        'an insect robot preparing a delicious meal', num_inference_steps=1)

    assert result['samples'].shape == (3, 512, 512)


if __name__ == '__main__':
    test_stable_diffusion()
