# Copyright (c) OpenMMLab. All rights reserved.
import gc
import platform
from copy import deepcopy
from unittest import TestCase

import pytest
import torch

from mmagic.models.editors.consistency_models.consistencymodel_utils import (
    DeterministicGenerator, DeterministicIndividualGenerator, DummyGenerator,
    get_generator, get_sample_fn, karras_sample, sample_dpm, sample_euler,
    sample_euler_ancestral, sample_heun, sample_onestep, sample_progdist,
    stochastic_iterative_sampler)
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
    image_size=64,
    out_channels=3,
    attention_resolutions=(2, 4, 8),
    in_channels=3,
    model_channels=192,
    num_res_blocks=3,
    dropout=0.0,
    channel_mult=(1, 2, 3, 4),
    use_checkpoint=False,
    use_fp16=False,
    num_head_channels=64,
    num_heads=4,
    num_heads_upsample=-1,
    resblock_updown=True,
    use_new_attention_order=False,
    use_scale_shift_norm=True)


@pytest.mark.skipif(
    'win' in platform.system().lower(),
    reason='skip on windows due to limited RAM.')
class TestConsistencyModelUtils(TestCase):

    def test_karras_sample(self):
        unet_cfg = deepcopy(unet_config)
        diffuse_cfg = deepcopy(denoiser_config)
        unet = MODELS.build(unet_cfg)
        diffuse = MODELS.build(diffuse_cfg)
        image_size = 64
        channel_num = 3
        steps = 2
        batch_size = 4
        model_kwargs = {}
        sample = karras_sample(
            diffuse,
            unet, (batch_size, channel_num, image_size, image_size),
            steps=steps,
            model_kwargs=model_kwargs)
        assert sample.shape == (batch_size, channel_num, image_size,
                                image_size)

    def test_get_generator(self):
        self.assertIsInstance(get_generator('dummy'), DummyGenerator)
        self.assertIsInstance(get_generator('determ'), DeterministicGenerator)
        self.assertIsInstance(
            get_generator('determ-indiv'), DeterministicIndividualGenerator)
        with pytest.raises(NotImplementedError):
            get_generator('')

    def test_sample_fn(self):
        self.assertEqual(get_sample_fn('heun'), sample_heun)
        self.assertEqual(get_sample_fn('dpm'), sample_dpm)
        self.assertEqual(get_sample_fn('ancestral'), sample_euler_ancestral)
        self.assertEqual(get_sample_fn('onestep'), sample_onestep)
        self.assertEqual(get_sample_fn('progdist'), sample_progdist)
        self.assertEqual(get_sample_fn('euler'), sample_euler)
        self.assertEqual(
            get_sample_fn('multistep'), stochastic_iterative_sampler)
        with pytest.raises(KeyError):
            get_sample_fn('')


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
