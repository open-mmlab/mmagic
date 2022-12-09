# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from unittest import TestCase

import pytest
import torch
import torch.nn as nn

from mmedit.models import DDIMScheduler, DenoisingUnet, DiscoDiffusion
from mmedit.utils import register_all_modules

register_all_modules()


class clip_mock(nn.Module):

    def __init__(self, device='cuda'):
        super().__init__()
        self.register_buffer('tensor', torch.randn([1, 512]))

    def encode_image(self, inputs):
        return inputs.mean() * self.tensor.repeat(inputs.shape[0], 1).to(
            inputs.device)

    def encode_text(self, inputs):
        return self.tensor.repeat(inputs.shape[0], 1).to(inputs.device)

    def forward(self, x):
        return x


class clip_mock_wrapper(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = clip_mock()

    def forward(self, x):
        return x


class TestDiscoDiffusion(TestCase):

    def setUp(self):
        # unet
        self.unet32 = DenoisingUnet(
            image_size=32,
            in_channels=3,
            base_channels=8,
            resblocks_per_downsample=2,
            attention_res=(8, ),
            norm_cfg=dict(type='GN32', num_groups=8),
            dropout=0.0,
            num_classes=0,
            use_fp16=True,
            resblock_updown=True,
            attention_cfg=dict(
                type='MultiHeadAttentionBlock',
                num_heads=2,
                num_head_channels=8,
                use_new_attention_order=False),
            use_scale_shift_norm=True)
        # mock clip
        self.clip_models = [clip_mock_wrapper(), clip_mock_wrapper()]
        # diffusion_scheduler
        self.diffusion_scheduler = DDIMScheduler(
            variance_type='learned_range',
            beta_schedule='linear',
            clip_sample=False)

    def test_init(self):
        unet32 = deepcopy(self.unet32)
        diffusion_scheduler = deepcopy(self.diffusion_scheduler)
        clip_models = deepcopy(self.clip_models)
        self.disco_diffusion = DiscoDiffusion(
            unet=unet32,
            diffusion_scheduler=diffusion_scheduler,
            secondary_model=None,
            clip_models=clip_models,
            use_fp16=True)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_infer(self):
        unet32 = deepcopy(self.unet32)
        diffusion_scheduler = deepcopy(self.diffusion_scheduler)
        clip_models = deepcopy(self.clip_models)
        self.disco_diffusion = DiscoDiffusion(
            unet=unet32,
            diffusion_scheduler=diffusion_scheduler,
            secondary_model=None,
            clip_models=clip_models,
            use_fp16=True)
        self.disco_diffusion.cuda().eval()

        # test model structure
        text_prompts = {
            0: ['clouds surround the mountains and palaces,sunshine,lake']
        }
        image = self.disco_diffusion.infer(
            text_prompts=text_prompts,
            show_progress=True,
            num_inference_steps=5,
            eta=0.8)['samples']
        assert image.shape == (1, 3, 32, 32)
        # test with different text prompts
        text_prompts = {
            0: [
                'a portrait of supergirl, by artgerm, rosstran, trending on artstation.'  # noqa
            ]
        }
        image = self.disco_diffusion.infer(
            text_prompts=text_prompts,
            show_progress=True,
            num_inference_steps=5,
            eta=0.8)['samples']
        assert image.shape == (1, 3, 32, 32)

        # test with init_image
        init_image = 'tests/data/image/face/000001.png'
        text_prompts = {
            0: [
                'a portrait of supergirl, by artgerm, rosstran, trending on artstation.'  # noqa
            ]
        }
        image = self.disco_diffusion.infer(
            text_prompts=text_prompts,
            init_image=init_image,
            show_progress=True,
            num_inference_steps=5,
            eta=0.8)['samples']
        assert image.shape == (1, 3, 32, 32)

        # test with different image resolution
        text_prompts = {
            0: ['clouds surround the mountains and palaces,sunshine,lake']
        }
        image = self.disco_diffusion.infer(
            height=64,
            width=128,
            text_prompts=text_prompts,
            show_progress=True,
            num_inference_steps=5,
            eta=0.8)['samples']
        assert image.shape == (1, 3, 64, 128)

        # clip guidance scale
        image = self.disco_diffusion.infer(
            text_prompts=text_prompts,
            show_progress=True,
            num_inference_steps=5,
            clip_guidance_scale=8000,
            eta=0.8)['samples']
        assert image.shape == (1, 3, 32, 32)

        # test with different loss settings
        tv_scale = 0.5
        sat_scale = 0.5
        range_scale = 100
        image = self.disco_diffusion.infer(
            text_prompts=text_prompts,
            show_progress=True,
            num_inference_steps=5,
            eta=0.8,
            tv_scale=tv_scale,
            sat_scale=sat_scale,
            range_scale=range_scale)['samples']
        assert image.shape == (1, 3, 32, 32)

        # test with different cutter settings
        cut_overview = [12] * 100 + [4] * 900
        cut_innercut = [4] * 100 + [12] * 900
        cut_ic_pow = [1] * 200 + [0] * 800
        cut_icgray_p = [0.2] * 200 + [0] * 800
        cutn_batches = 2
        image = self.disco_diffusion.infer(
            text_prompts=text_prompts,
            show_progress=True,
            num_inference_steps=5,
            eta=0.8,
            cut_overview=cut_overview,
            cut_innercut=cut_innercut,
            cut_ic_pow=cut_ic_pow,
            cut_icgray_p=cut_icgray_p,
            cutn_batches=cutn_batches)['samples']
        assert image.shape == (1, 3, 32, 32)

        # test with different unet
        unet64 = DenoisingUnet(
            image_size=64,
            in_channels=3,
            base_channels=8,
            resblocks_per_downsample=2,
            attention_res=(8, ),
            norm_cfg=dict(type='GN32', num_groups=8),
            dropout=0.0,
            num_classes=0,
            use_fp16=True,
            resblock_updown=True,
            attention_cfg=dict(
                type='MultiHeadAttentionBlock',
                num_heads=2,
                num_head_channels=8,
                use_new_attention_order=False),
            use_scale_shift_norm=True).cuda()
        unet64.convert_to_fp16()
        self.disco_diffusion.unet = unet64
        image = self.disco_diffusion.infer(
            text_prompts=text_prompts,
            show_progress=True,
            num_inference_steps=5,
            eta=0.8)['samples']
        assert image.shape == (1, 3, 64, 64)
