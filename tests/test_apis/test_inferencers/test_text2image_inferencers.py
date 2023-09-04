# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import unittest
from unittest import TestCase
from unittest.mock import patch

import torch
import torch.nn as nn
from mmengine.utils import digit_version
from torchvision.version import __version__ as TV_VERSION

from mmagic.apis.inferencers.text2image_inferencer import Text2ImageInferencer
from mmagic.models import DenoisingUnet, DiscoDiffusion
from mmagic.models.diffusion_schedulers import EditDDIMScheduler
from mmagic.utils import register_all_modules

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


class TestTranslationInferencer(TestCase):

    def setUp(self):
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
        self.diffusion_scheduler = EditDDIMScheduler(
            variance_type='learned_range',
            beta_schedule='linear',
            clip_sample=False)

        unet32 = self.unet32
        diffusion_scheduler = self.diffusion_scheduler
        clip_models = self.clip_models
        self.disco_diffusion = DiscoDiffusion(
            unet=unet32,
            diffusion_scheduler=diffusion_scheduler,
            secondary_model=None,
            clip_models=clip_models,
            use_fp16=True).cuda()

    @unittest.skipIf(
        digit_version(TV_VERSION) <= digit_version('0.7.0'),
        reason='torchvision version limiation')
    @unittest.skipIf(not torch.cuda.is_available(), reason='requires cuda')
    def test_translation(self):
        cfg_root = osp.join(
            osp.dirname(__file__), '..', '..', '..', 'configs',
            'disco_diffusion')
        cfg = osp.join(cfg_root,
                       'disco-diffusion_adm-u-finetuned_imagenet-512x512.py')
        text = {0: ['sad']}
        result_out_dir = osp.join(
            osp.dirname(__file__), '..', '..', 'data/out', 'disco_result.png')

        with patch.object(Text2ImageInferencer, '_init_model'):
            inferencer_instance = Text2ImageInferencer(
                cfg, None, extra_parameters={'num_inference_steps': 2})
            # mock model
            inferencer_instance.model = self.disco_diffusion
            inferencer_instance(text=text)
            inference_result = inferencer_instance(
                text=text, result_out_dir=result_out_dir)
            result_img = inference_result[1]
            assert result_img[0].cpu().numpy().shape == (3, 32, 32)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
