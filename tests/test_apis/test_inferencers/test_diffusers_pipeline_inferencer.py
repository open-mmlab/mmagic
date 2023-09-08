# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION

from mmagic.apis.inferencers.diffusers_pipeline_inferencer import \
    DiffusersPipelineInferencer
from mmagic.utils import register_all_modules

register_all_modules()


@pytest.mark.skipif(
    'win' in platform.system().lower()
    or digit_version(TORCH_VERSION) <= digit_version('1.8.1'),
    reason='skip on windows due to limited RAM'
    'and get_submodule requires torch >= 1.9.0')
def test_diffusers_pipeline_inferencer():
    cfg = dict(
        model=dict(
            type='DiffusionPipeline',
            from_pretrained='runwayml/stable-diffusion-v1-5'))

    inferencer_instance = DiffusersPipelineInferencer(cfg, None)
    text_prompts = 'Japanese anime style, girl'
    negative_prompt = 'bad face, bad hands'
    result = inferencer_instance(
        text=text_prompts,
        negative_prompt=negative_prompt,
        height=64,
        width=64)
    assert result[1][0].size == (64, 64)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
