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
            type='DiffusionPipeline', from_pretrained='google/ddpm-cat-256'))

    inferencer_instance = DiffusersPipelineInferencer(cfg, None)
    result = inferencer_instance()
    assert result[1][0].size == (256, 256)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
