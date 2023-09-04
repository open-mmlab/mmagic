# Copyright (c) OpenMMLab. All rights reserved.
import platform
import unittest
from unittest import TestCase

import torch

from mmagic.models.diffusion_schedulers import EditDDIMScheduler
from mmagic.utils import register_all_modules
from projects.glide.models import Glide, SuperResText2ImUNet, Text2ImUNet

register_all_modules()


class TestGLIDE(TestCase):

    def setUp(self):
        # low resolution cfg
        unet_cfg = dict(
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
        diffusion_scheduler_cfg = dict(
            variance_type='learned_range', beta_schedule='squaredcos_cap_v2')

        # unet
        self.unet = Text2ImUNet(**unet_cfg)
        # diffusion_scheduler
        self.diffusion_scheduler = EditDDIMScheduler(**diffusion_scheduler_cfg)

        # high resolution cfg
        unet_up_cfg = dict(
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
        diffusion_scheduler_up_cfg = dict(
            variance_type='learned_range', beta_schedule='linear')

        # unet up
        self.unet_up = SuperResText2ImUNet(**unet_up_cfg)
        self.diffusion_scheduler_up = EditDDIMScheduler(
            **diffusion_scheduler_up_cfg)

    @unittest.skipIf(
        'win' in platform.system().lower(),
        reason='skip on windows due to limited RAM.')
    def test_init(self):
        self.GLIDE = Glide(
            unet=self.unet,
            diffusion_scheduler=self.diffusion_scheduler,
            unet_up=self.unet_up,
            diffusion_scheduler_up=self.diffusion_scheduler_up)

    @unittest.skipIf(
        ('win' in platform.system().lower())
        or (not torch.cuda.is_available()),
        reason='skip on windows and cpu due to limited RAM.')
    def test_infer(self):
        # test infer resolution
        text_prompts = 'clouds surround the mountains and palaces,sunshine'
        image = self.GLIDE.infer(
            prompt=text_prompts, show_progress=True,
            num_inference_steps=2)['samples']
        assert image.shape == (1, 3, 256, 256)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
