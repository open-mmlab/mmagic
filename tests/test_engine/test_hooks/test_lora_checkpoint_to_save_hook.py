# Copyright (c) OpenMMLab. All rights reserved.
import copy
import gc
import platform

import pytest
import torch
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from mmengine import Config
from mmengine.registry import MODELS
from mmengine.testing import RunnerTestCase

from mmagic.engine.hooks import LoRACheckpointToSaveHook
from mmagic.models.archs import DiffusersWrapper
from mmagic.models.data_preprocessors import DataPreprocessor
from mmagic.models.editors import ClipWrapper, StableDiffusionXL
from mmagic.models.editors.stable_diffusion import AutoencoderKL

stable_diffusion_xl_tiny_url = 'hf-internal-testing/tiny-stable-diffusion-xl-pipe'  # noqa
lora_config = dict(target_modules=['to_q', 'to_k', 'to_v'])
diffusion_scheduler = dict(
    type='EditDDIMScheduler',
    variance_type='learned_range',
    beta_end=0.012,
    beta_schedule='scaled_linear',
    beta_start=0.00085,
    num_train_timesteps=1000,
    set_alpha_to_one=False,
    clip_sample=False)
model = dict(
    type='StableDiffusionXL',
    unet=dict(
        type='UNet2DConditionModel',
        subfolder='unet',
        from_pretrained=stable_diffusion_xl_tiny_url),
    vae=dict(type='AutoencoderKL', sample_size=64),
    text_encoder_one=dict(
        type='ClipWrapper',
        clip_type='huggingface',
        pretrained_model_name_or_path=stable_diffusion_xl_tiny_url,
        subfolder='text_encoder'),
    tokenizer_one=stable_diffusion_xl_tiny_url,
    text_encoder_two=dict(
        type='ClipWrapper',
        clip_type='huggingface',
        pretrained_model_name_or_path=stable_diffusion_xl_tiny_url,
        subfolder='text_encoder_2'),
    tokenizer_two=stable_diffusion_xl_tiny_url,
    scheduler=diffusion_scheduler,
    val_prompts=['a dog', 'a dog'],
    lora_config=lora_config)


@pytest.mark.skipif(
    'win' in platform.system().lower(),
    reason='skip on windows due to limited RAM.')
@pytest.mark.skipif(
    torch.__version__ < '1.9.0',
    reason='skip on torch<1.9 due to unsupported torch.concat')
class TestLoRACheckpointToSaveHook(RunnerTestCase):

    def setUp(self) -> None:
        MODELS.register_module(
            name='StableDiffusionXL', module=StableDiffusionXL)
        MODELS.register_module(name='ClipWrapper', module=ClipWrapper)

        def gen_wrapped_cls(module, module_name):
            return type(
                module_name, (DiffusersWrapper, ),
                dict(
                    _module_cls=module,
                    _module_name=module_name,
                    __module__=__name__))

        wrapped_module = gen_wrapped_cls(UNet2DConditionModel,
                                         'UNet2DConditionModel')
        MODELS.register_module(
            name='UNet2DConditionModel', module=wrapped_module, force=True)
        MODELS.register_module(name='AutoencoderKL', module=AutoencoderKL)
        MODELS.register_module(
            name='DataPreprocessor', module=DataPreprocessor)
        return super().setUp()

    def tearDown(self):
        MODELS.module_dict.pop('StableDiffusionXL')
        MODELS.module_dict.pop('ClipWrapper')
        MODELS.module_dict.pop('UNet2DConditionModel')
        MODELS.module_dict.pop('AutoencoderKL')
        MODELS.module_dict.pop('DataPreprocessor')

        gc.collect()
        globals().clear()
        locals().clear()

        return super().tearDown()

    def test_init(self):
        LoRACheckpointToSaveHook()

    def test_before_save_checkpoint(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        runner = self.build_runner(cfg)
        runner.model = MODELS.build(Config(model))
        checkpoint = dict(state_dict=MODELS.build(Config(model)).state_dict())
        hook = LoRACheckpointToSaveHook()
        hook.before_save_checkpoint(runner, checkpoint)

        for key in checkpoint['state_dict'].keys():
            assert 'lora_mapping' in key


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
