# Copyright (c) OpenMMLab. All rights reserved.
import gc
import platform
from copy import deepcopy
from unittest import TestCase

import pytest
import torch

from mmagic.models import (ConsistencyModel, ConsistencyUNetModel,
                           DataPreprocessor, KarrasDenoiser)
from mmagic.registry import MODELS
from mmagic.utils import register_all_modules

gc.collect()
torch.cuda.empty_cache()
register_all_modules()
denoiser_config = dict(
    type='KarrasDenoiser',
    sigma_data=0.5,
    sigma_max=80.0,
    sigma_min=0.002,
    weight_schedule='uniform',
)

unet_config = dict(
    type='ConsistencyUNetModel',
    in_channels=3,
    model_channels=192,
    num_res_blocks=3,
    dropout=0.0,
    channel_mult='',
    use_checkpoint=False,
    use_fp16=False,
    num_head_channels=64,
    num_heads=4,
    num_heads_upsample=-1,
    resblock_updown=True,
    use_new_attention_order=False,
    use_scale_shift_norm=True)

config_onestep = dict(
    type='ConsistencyModel',
    unet=unet_config,
    denoiser=denoiser_config,
    attention_resolutions='32,16,8',
    batch_size=4,
    class_cond=True,
    generator='determ',
    image_size=64,
    learn_sigma=False,
    model_path=None,
    num_classes=1000,
    sampler='onestep',
    seed=42,
    training_mode='consistency_distillation',
    ts='',
    data_preprocessor=dict(
        type='DataPreprocessor', mean=[127.5] * 3, std=[127.5] * 3))

config_multistep = dict(
    type='ConsistencyModel',
    unet=unet_config,
    denoiser=denoiser_config,
    attention_resolutions='32,16,8',
    batch_size=4,
    class_cond=True,
    generator='determ',
    image_size=64,
    learn_sigma=False,
    model_path=None,
    num_classes=1000,
    sampler='multistep',
    seed=42,
    steps=40,
    training_mode='consistency_distillation',
    ts='0,22,39',
    data_preprocessor=dict(
        type='DataPreprocessor', mean=[127.5] * 3, std=[127.5] * 3))


@pytest.mark.skipif(
    'win' in platform.system().lower(),
    reason='skip on windows due to limited RAM.')
class TestDeblurGanV2(TestCase):

    def test_init(self):
        model = ConsistencyModel(
            unet=unet_config,
            denoiser=denoiser_config,
            data_preprocessor=DataPreprocessor())
        self.assertIsInstance(model, ConsistencyModel)
        self.assertIsInstance(model.data_preprocessor, DataPreprocessor)
        self.assertIsInstance(model.model, ConsistencyUNetModel)
        self.assertIsInstance(model.diffusion, KarrasDenoiser)
        unet_cfg = deepcopy(unet_config)
        diffuse_cfg = deepcopy(denoiser_config)
        unet = MODELS.build(unet_cfg)
        diffuse = MODELS.build(diffuse_cfg)
        model = ConsistencyModel(
            unet=unet, denoiser=diffuse, data_preprocessor=DataPreprocessor())
        self.assertIsInstance(model.model, ConsistencyUNetModel)
        self.assertIsInstance(model.diffusion, KarrasDenoiser)

    def test_onestep_infer(self):
        model = MODELS.build(config_onestep)
        data = {
            'num_batches': model.batch_size,
            'labels': None,
            'sample_model': 'orig'
        }
        result = model(data)
        assert len(result) == model.batch_size
        for datasample in result:
            assert datasample.fake_img.shape == (3, model.image_size,
                                                 model.image_size)

    def test_multistep_infer(self):
        model = MODELS.build(config_multistep)
        data = {
            'num_batches': model.batch_size,
            'labels': None,
            'sample_model': 'orig'
        }
        result = model(data)
        assert len(result) == model.batch_size
        for datasample in result:
            assert datasample.fake_img.shape == (3, model.image_size,
                                                 model.image_size)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
