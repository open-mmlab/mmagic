# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch
from mmengine import MODELS, Config
from mmengine.optim import OptimWrapper
from torch.optim import SGD

from mmagic.structures import DataSample
from mmagic.utils import register_all_modules

register_all_modules()

stable_diffusion_xl_tiny_url = 'hf-internal-testing/tiny-stable-diffusion-xl-pipe'  # noqa
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
    scheduler=dict(
        type='DDPMScheduler',
        from_pretrained=stable_diffusion_xl_tiny_url,
        subfolder='scheduler'),
    test_scheduler=dict(
        type='DDIMScheduler',
        from_pretrained=stable_diffusion_xl_tiny_url,
        subfolder='scheduler'),
    val_prompts=['a dog', 'a dog'])


@pytest.mark.skipif(
    'win' in platform.system().lower(),
    reason='skip on windows due to limited RAM.')
@pytest.mark.skipif(
    torch.__version__ < '1.9.0',
    reason='skip on torch<1.9 due to unsupported torch.concat')
def test_stable_xl_diffusion():
    StableDiffuser = MODELS.build(Config(model))

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


@pytest.mark.skipif(
    'win' in platform.system().lower(),
    reason='skip on windows due to limited RAM.')
@pytest.mark.skipif(
    torch.__version__ < '1.9.0',
    reason='skip on torch<1.9 due to unsupported torch.concat')
def test_stable_diffusion_xl_step():
    StableDiffuser = MODELS.build(Config(model))

    # train step
    data = dict(
        inputs=torch.ones([1, 3, 64, 64]),
        time_ids=torch.zeros((1, 6)),
        data_samples=[
            DataSample(prompt='an insect robot preparing a delicious meal')
        ])
    optimizer = SGD(StableDiffuser.parameters(), lr=0.1)
    optim_wrapper = OptimWrapper(optimizer)
    log_vars = StableDiffuser.train_step(data, optim_wrapper)
    assert log_vars
    assert isinstance(log_vars['loss'], torch.Tensor)

    # val step
    data = dict(data_samples=[DataSample()])
    outputs = StableDiffuser.val_step(data)
    assert len(outputs) == 2
    assert isinstance(outputs[0].fake_img, torch.Tensor)
    assert outputs[0].fake_img.shape == (3, 32, 32)

    # test step
    outputs = StableDiffuser.test_step(data)
    assert len(outputs) == 2
    assert isinstance(outputs[0].fake_img, torch.Tensor)
    assert outputs[0].fake_img.shape == (3, 32, 32)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
