# Copyright (c) OpenMMLab. All rights reserved.
import platform
from unittest import TestCase

import pytest
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION

from mmagic.registry import MODELS
from mmagic.utils import register_all_modules

# from mmengine.utils import digit_version

# from mmengine.utils.dl_utils import TORCH_VERSION

register_all_modules()

stable_diffusion_v15_url = 'runwayml/stable-diffusion-v1-5'

diffusion_scheduler = dict(
    type='DDIMScheduler',
    beta_end=0.012,
    beta_schedule='linear',
    beta_start=0.00085,
    num_train_timesteps=1000,
    prediction_type='epsilon',
    set_alpha_to_one=True,
    clip_sample=False,
    thresholding=False,
    steps_offset=1)

model = dict(
    type='AnimateDiff',
    vae=dict(
        type='AutoencoderKL',
        from_pretrained=stable_diffusion_v15_url,
        subfolder='vae'),
    unet=dict(
        type='UNet3DConditionMotionModel',
        unet_use_cross_frame_attention=False,
        unet_use_temporal_attention=False,
        use_motion_module=True,
        motion_module_resolutions=[1, 2, 4, 8],
        motion_module_mid_block=False,
        motion_module_decoder_only=False,
        motion_module_type='Vanilla',
        motion_module_kwargs=dict(
            num_attention_heads=8,
            num_transformer_block=1,
            attention_block_types=['Temporal_Self', 'Temporal_Self'],
            temporal_position_encoding=True,
            temporal_position_encoding_max_len=24,
            temporal_attention_dim_div=1)),
    text_encoder=dict(
        type='ClipWrapper',
        clip_type='huggingface',
        pretrained_model_name_or_path=stable_diffusion_v15_url,
        subfolder='text_encoder'),
    tokenizer=stable_diffusion_v15_url,
    scheduler=diffusion_scheduler,
    test_scheduler=diffusion_scheduler,
    data_preprocessor=dict(type='DataPreprocessor'),
    dream_booth_lora_cfg=dict(type='ToonYou', steps=25, guidance_scale=7.5))


@pytest.mark.skipif(
    'win' in platform.system().lower()
    or digit_version(TORCH_VERSION) <= digit_version('1.9.2'),
    reason='skip on windows due to limited RAM'
    'and torch >= 1.10.0')
class TestAnimateDiff(TestCase):

    def setUp(self):
        animatediff = MODELS.build(model)
        assert not any([p.requires_grad for p in animatediff.vae.parameters()])
        assert not any(
            [p.requires_grad for p in animatediff.text_encoder.parameters()])
        assert not any(
            [p.requires_grad for p in animatediff.unet.parameters()])
        self.animatediff = animatediff

    @pytest.mark.skipif(
        'win' in platform.system().lower(),
        reason='skip on windows due to limited RAM.')
    def test_infer(self):
        videos = self.animatediff.infer(
            'best quality, masterpiece, 1girl, cloudy sky, \
            dandelion, contrapposto, alternate hairstyle',
            negative_prompt='',
            video_length=16,
            height=32,
            width=32)['samples']
        assert videos.shape == (1, 3, 16, 32, 32)
