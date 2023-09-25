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
val_prompts = ['a photo of S*']
image_cross_layers = [
    # down blocks (2x transformer block) * (3x down blocks) = 6
    0,
    0,
    0,
    0,
    0,
    0,
    # mid block (1x transformer block) * (1x mid block)= 1
    0,
    # up blocks (3x transformer block) * (3x up blocks) = 9
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
]
reg_loss_weight: float = 5e-4
placeholder: str = 'S*'
initialize_token: str = 'dog'
num_vectors_per_token: int = 1

config = dict(
    type='ViCo',
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
    image_cross_layers=image_cross_layers,
    reg_loss_weight=reg_loss_weight,
    placeholder=placeholder,
    initialize_token=initialize_token,
    num_vectors_per_token=num_vectors_per_token,
    val_prompts=val_prompts)


@pytest.mark.skipif(
    'win' in platform.system().lower()
    or digit_version(TORCH_VERSION) <= digit_version('1.8.1'),
    reason='skip on windows due to limited RAM'
    'and get_submodule requires torch >= 1.9.0')
class TestViCo(TestCase):

    def setUp(self):
        # mock SiLU
        if digit_version(TORCH_VERSION) <= digit_version('1.6.0'):
            from mmagic.models.editors.ddpm.denoising_unet import SiLU
            torch.nn.SiLU = SiLU
        vico = MODELS.build(config)
        assert not any([p.requires_grad for p in vico.vae.parameters()])
        self.vico = vico

    def test_infer(self):
        vico = self.vico

        def mock_encode_prompt(prompt, do_classifier_free_guidance,
                               num_images_per_prompt, *args, **kwargs):
            batch_size = len(prompt) if isinstance(prompt, list) else 1
            batch_size *= num_images_per_prompt
            if do_classifier_free_guidance:
                batch_size *= 2
            return torch.randn(batch_size, 5, 16)  # 2 for cfg

        def mock_infer(prompt, *args, **kwargs):
            length = len(prompt)
            return dict(samples=torch.randn(length, 3, 64, 64))

        encode_prompt = vico._encode_prompt
        infer = vico.infer
        vico._encode_prompt = mock_encode_prompt
        vico.infer = mock_infer

        self._test_infer(vico, 1, 1)
        vico._encode_prompt = encode_prompt
        vico.infer = infer

    def _test_infer(self, vico, num_prompt, num_repeat):
        prompt = ''
        image_reference = torch.ones([1, 3, 64, 64])

        result = vico.infer(
            [prompt] * num_prompt,
            image_reference=image_reference,
            height=64,
            width=64,
            num_images_per_prompt=num_repeat,
            num_inference_steps=1,
            return_type='numpy')
        assert result['samples'].shape == (1, 3, 64, 64)

    def test_val_step(self):
        vico = self.vico
        data = dict(
            inputs=[
                dict(
                    img=torch.ones([3, 64, 64]),
                    img_ref=torch.ones([3, 64, 64]))
            ],
            data_samples=[DataSample(prompt='a photo of S*')])

        def mock_encode_prompt(*args, **kwargs):
            return torch.randn(2, 5, 16)  # 2 for cfg

        def mock_infer(prompt, image_reference, *args, **kwargs):
            length = len(prompt)
            return dict(samples=torch.randn(length, 3, 64, 64))

        encode_prompt = vico._encode_prompt
        infer = vico.infer
        vico._encode_prompt = mock_encode_prompt
        vico.infer = mock_infer

        output = vico.val_step(data)
        assert len(output) == 1
        vico._encode_prompt = encode_prompt
        vico.infer = infer

    def test_test_step(self):
        vico = self.vico
        data = dict(
            inputs=[
                dict(
                    img=torch.ones([3, 64, 64]),
                    img_ref=torch.ones([3, 64, 64]))
            ],
            data_samples=[DataSample(prompt='a photo of S*')])

        def mock_encode_prompt(*args, **kwargs):
            return torch.randn(2, 5, 16)  # 2 for cfg

        def mock_infer(prompt, image_reference, *args, **kwargs):
            length = len(prompt)
            return dict(samples=torch.randn(length, 3, 64, 64))

        encode_prompt = vico._encode_prompt
        infer = vico.infer
        vico._encode_prompt = mock_encode_prompt
        vico.infer = mock_infer

        output = vico.test_step(data)
        assert len(output) == 1
        vico._encode_prompt = encode_prompt
        vico.infer = infer

    def test_train_step(self):
        vico = self.vico
        data = dict(
            inputs=[
                dict(
                    img=torch.ones([3, 64, 64]),
                    img_ref=torch.ones([3, 64, 64]))
            ],
            data_samples=[DataSample(prompt='a photo of S*')])

        optimizer = MagicMock()
        update_params = MagicMock()
        optimizer.update_params = update_params
        optim_wrapper = optimizer

        class mock_text_encoder(nn.Module):

            def __init__(self):
                super().__init__()

            def forward(self, *args, **kwargs):
                return [torch.randn(1, 5, 16)]

        vico.text_encoder = mock_text_encoder()

        vico.train_step(data, optim_wrapper)
