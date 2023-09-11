# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch
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

    def mock_encode_prompt(prompt, do_classifier_free_guidance,
                           num_images_per_prompt, *args, **kwargs):
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        batch_size *= num_images_per_prompt
        if do_classifier_free_guidance:
            batch_size *= 2
        return torch.randn(batch_size, 5, 16)  # 2 for cfg

    inferencer_instance.model._encode_prompt = mock_encode_prompt

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
