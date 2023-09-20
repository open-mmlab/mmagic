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

stable_diffusion_tiny_url = 'hf-internal-testing/tiny-stable-diffusion-pipe'
config = dict(
    type='ControlStableDiffusion',
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
    controlnet=dict(
        type='ControlNetModel',
        # from_pretrained=controlnet_canny_rul
        from_config=config_path  # train from scratch
    ),
    scheduler=dict(
        type='DDPMScheduler',
        from_pretrained=stable_diffusion_tiny_url,
        subfolder='scheduler'),
    test_scheduler=dict(
        type='DDIMScheduler',
        from_pretrained=stable_diffusion_tiny_url,
        subfolder='scheduler'),
    data_preprocessor=dict(type='DataPreprocessor'),
    enable_xformers=False,
    init_cfg=dict(type='init_from_unet'))


@pytest.mark.skipif(
    'win' in platform.system().lower(),
    reason='skip on windows due to limited RAM.')
class TestControlStableDiffusion(TestCase):

    def setUp(self):
        # mock SiLU
        if digit_version(TORCH_VERSION) <= digit_version('1.6.0'):
            from mmagic.models.editors.ddpm.denoising_unet import SiLU
            torch.nn.SiLU = SiLU
        control_sd = MODELS.build(config)
        assert not any([p.requires_grad for p in control_sd.vae.parameters()])
        assert not any(
            [p.requires_grad for p in control_sd.text_encoder.parameters()])
        assert not any([p.requires_grad for p in control_sd.unet.parameters()])
        self.control_sd = control_sd

    def test_init_weights(self):
        control_sd = self.control_sd
        # test init_from_unet
        control_sd.init_weights()

        # test init_convert_from_unet
        unet = dict(
            type='UNet2DConditionModel',
            down_block_types=('DownBlock2D', ),
            up_block_types=('UpBlock2D', ),
            block_out_channels=(32, ),
            cross_attention_dim=16)
        control_sd.init_cfg = dict(type='convert_from_unet', base_model=unet)
        control_sd.init_weights()

    def test_infer(self):
        control_sd = self.control_sd

        def mock_encode_prompt(prompt, do_classifier_free_guidance,
                               num_images_per_prompt, *args, **kwargs):
            batch_size = len(prompt) if isinstance(prompt, list) else 1
            batch_size *= num_images_per_prompt
            if do_classifier_free_guidance:
                batch_size *= 2
            return torch.randn(batch_size, 5, 16)  # 2 for cfg

        encode_prompt = control_sd._encode_prompt
        control_sd._encode_prompt = mock_encode_prompt

        # one prompt, one control, repeat 1 time
        self._test_infer(control_sd, 1, 1, 1, 1)

        # two prompt, two control, repeat 1 time
        # NOTE: skip this due to memory limit
        # self._test_infer(control_sd, 2, 2, 1, 2)

        # one prompt, one control, repeat 2 times
        # NOTE: skip this due to memory limit
        # self._test_infer(control_sd, 1, 1, 2, 2)

        # two prompt, two control, repeat 2 times
        # NOTE: skip this due to memory limit
        # self._test_infer(control_sd, 2, 2, 2, 4)

        control_sd._encode_prompt = encode_prompt

    def _test_infer(self, control_sd, num_prompt, num_control, num_repeat,
                    tar_shape):
        prompt = ''
        control = torch.ones([1, 3, 64, 64])

        result = control_sd.infer(
            [prompt] * num_prompt,
            control=[control] * num_control,
            height=64,
            width=64,
            num_images_per_prompt=num_repeat,
            num_inference_steps=1,
            return_type='numpy')
        assert result['samples'].shape == (tar_shape, 3, 64, 64)

    def test_val_step(self):
        control_sd = self.control_sd
        data = dict(
            inputs=[
                dict(
                    target=torch.ones([3, 64, 64]),
                    source=torch.ones([3, 64, 64]))
            ],
            data_samples=[
                DataSample(prompt='an insect robot preparing a delicious meal')
            ])

        def mock_encode_prompt(*args, **kwargs):
            return torch.randn(2, 5, 16)  # 2 for cfg

        def mock_infer(*args, **kwargs):
            return dict(samples=torch.randn(2, 3, 64, 64))

        encode_prompt = control_sd._encode_prompt
        control_sd._encode_prompt = mock_encode_prompt

        infer = control_sd.infer
        control_sd.infer = mock_infer

        # control_sd.text_encoder = mock_text_encoder()
        output = control_sd.val_step(data)
        assert len(output) == 1
        control_sd._encode_prompt = encode_prompt
        control_sd.infer = infer

    def test_test_step(self):
        control_sd = self.control_sd
        data = dict(
            inputs=[
                dict(
                    target=torch.ones([3, 64, 64]),
                    source=torch.ones([3, 64, 64]))
            ],
            data_samples=[
                DataSample(prompt='an insect robot preparing a delicious meal')
            ])

        def mock_encode_prompt(*args, **kwargs):
            return torch.randn(2, 5, 16)  # 2 for cfg

        def mock_infer(*args, **kwargs):
            return dict(samples=torch.randn(2, 3, 64, 64))

        encode_prompt = control_sd._encode_prompt
        control_sd._encode_prompt = mock_encode_prompt

        infer = control_sd.infer
        control_sd.infer = mock_infer

        # control_sd.text_encoder = mock_text_encoder()
        output = control_sd.test_step(data)
        assert len(output) == 1
        control_sd._encode_prompt = encode_prompt
        control_sd.infer = infer

    def test_train_step(self):
        control_sd = self.control_sd
        data = dict(
            inputs=[
                dict(
                    target=torch.ones([3, 64, 64]),
                    source=torch.ones([3, 64, 64]))
            ],
            data_samples=[
                DataSample(prompt='an insect robot preparing a delicious meal')
            ])

        optimizer = MagicMock()
        update_params = MagicMock()
        optimizer.update_params = update_params
        optim_wrapper = {'controlnet': optimizer}

        class mock_text_encoder(nn.Module):

            def __init__(self):
                super().__init__()

            def forward(self, *args, **kwargs):
                return [torch.randn(1, 5, 16)]

        control_sd.text_encoder = mock_text_encoder()

        control_sd.train_step(data, optim_wrapper)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
