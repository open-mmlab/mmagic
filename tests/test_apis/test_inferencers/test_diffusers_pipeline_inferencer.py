# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import platform
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
        height=128,
        width=128)
    assert result[1][0].size == (128, 128)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
