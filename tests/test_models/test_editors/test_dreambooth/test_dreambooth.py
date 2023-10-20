# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import platform
from unittest import TestCase
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION

from mmagic.registry import MODELS
from mmagic.structures import DataSample
from mmagic.utils import register_all_modules

test_dir = osp.join(osp.dirname(__file__), '../../../..', 'tests')
config_path = osp.join(test_dir, 'configs', 'diffuser_wrapper_cfg')
model_path = osp.join(test_dir, 'configs', 'tmp_weight')
ckpt_path = osp.join(test_dir, 'configs', 'ckpt')

register_all_modules()

stable_diffusion_tiny_url = 'diffusers/tiny-stable-diffusion-torch'
val_prompts = ['a sks dog in basket']

config = dict(
    type='DreamBooth',
    vae=dict(type='AutoencoderKL', sample_size=64),
    unet=dict(
        sample_size=64,
        type='UNet2DConditionModel',
        down_block_types=('DownBlock2D', ),
        up_block_types=('UpBlock2D', ),
        block_out_channels=(32, ),
        cross_attention_dim=16,
    ),
    text_encoder=dict(
        type='ClipWrapper',
        clip_type='huggingface',
        pretrained_model_name_or_path=stable_diffusion_tiny_url,
        subfolder='text_encoder'),
    tokenizer=stable_diffusion_tiny_url,
    scheduler=dict(
        type='DDPMScheduler',
        from_pretrained=stable_diffusion_tiny_url,
        subfolder='scheduler'),
    test_scheduler=dict(
        type='DDIMScheduler',
        from_pretrained=stable_diffusion_tiny_url,
        subfolder='scheduler'),
    dtype='fp32',
    data_preprocessor=dict(type='DataPreprocessor'),
    enable_xformers=False,
    val_prompts=val_prompts)


@pytest.mark.skipif(
    'win' in platform.system().lower(),
    reason='skip on windows due to limited RAM.')
class TestControlStableDiffusion(TestCase):

    def setUp(self):
        # mock SiLU
        if digit_version(TORCH_VERSION) <= digit_version('1.6.0'):
            from mmagic.models.editors.ddpm.denoising_unet import SiLU
            torch.nn.SiLU = SiLU
        dreambooth = MODELS.build(config)
        assert not any([p.requires_grad for p in dreambooth.vae.parameters()])
        assert not any(
            [p.requires_grad for p in dreambooth.text_encoder.parameters()])
        self.dreambooth = dreambooth

    def test_val_step(self):
        dreambooth = self.dreambooth
        data = dict(
            inputs=[torch.ones([3, 64, 64])],
            data_samples=[
                DataSample(prompt='an insect robot preparing a delicious meal')
            ])

        def mock_encode_prompt(*args, **kwargs):
            return torch.randn(2, 5, 16)  # 2 for cfg

        def mock_infer(prompt, *args, **kwargs):
            length = len(prompt)
            return dict(samples=torch.randn(length, 3, 64, 64))

        encode_prompt = dreambooth._encode_prompt
        infer = dreambooth.infer
        dreambooth._encode_prompt = mock_encode_prompt
        dreambooth.infer = mock_infer

        output = dreambooth.val_step(data)
        assert len(output) == 1
        dreambooth._encode_prompt = encode_prompt
        dreambooth.infer = infer

    def test_test_step(self):
        dreambooth = self.dreambooth
        data = dict(
            inputs=[torch.ones([3, 64, 64])],
            data_samples=[
                DataSample(prompt='an insect robot preparing a delicious meal')
            ])

        def mock_encode_prompt(*args, **kwargs):
            return torch.randn(2, 5, 16)  # 2 for cfg

        def mock_infer(prompt, *args, **kwargs):
            length = len(prompt)
            return dict(samples=torch.randn(length, 3, 64, 64))

        encode_prompt = dreambooth._encode_prompt
        infer = dreambooth.infer
        dreambooth._encode_prompt = mock_encode_prompt
        dreambooth.infer = mock_infer

        output = dreambooth.test_step(data)
        assert len(output) == 1
        dreambooth._encode_prompt = encode_prompt
        dreambooth.infer = infer

    def test_train_step(self):
        dreambooth = self.dreambooth
        data = dict(
            inputs=[torch.ones([3, 64, 64])],
            data_samples=[
                DataSample(prompt='an insect robot preparing a delicious meal')
            ])

        optimizer = MagicMock()
        update_params = MagicMock()
        optimizer.update_params = update_params
        optim_wrapper = optimizer

        class mock_text_encoder(nn.Module):

            def __init__(self):
                super().__init__()

            def forward(self, *args, **kwargs):
                return [torch.randn(1, 5, 16)]

        dreambooth.text_encoder = mock_text_encoder()

        dreambooth.train_step(data, optim_wrapper)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
