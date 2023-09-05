# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import sys

import pytest
import torch


def test_clip_wrapper():
    from transformers import CLIPConfig

    from mmagic.models.editors.stable_diffusion.clip_wrapper import \
        StableDiffusionSafetyChecker
    clipconfig = CLIPConfig()
    safety_checker = StableDiffusionSafetyChecker(clipconfig)

    clip_input = torch.rand((1, 3, 224, 224))
    images_input = torch.rand((1, 512, 512, 3))

    result = safety_checker.forward(clip_input, images_input)
    assert result[0].shape == (1, 512, 512, 3)


def test_load_clip_submodels():
    from mmagic.models.editors.stable_diffusion.clip_wrapper import \
        load_clip_submodels
    init_cfg = dict(
        type='Pretrained',
        pretrained_model_path='tem',
    )

    submodels = []
    with pytest.raises(Exception):
        load_clip_submodels(init_cfg, submodels, True)


def test_load_clip_submodels_transformers_none():
    transformer_location = sys.modules['transformers']
    sys.modules['transformers'] = None
    importlib.reload(
        sys.modules['mmagic.models.editors.stable_diffusion.clip_wrapper'])
    from mmagic.models.editors.stable_diffusion.clip_wrapper import \
        load_clip_submodels

    init_cfg = dict(
        type='Pretrained',
        pretrained_model_path='tem',
    )
    submodels = []
    with pytest.raises(ImportError):
        load_clip_submodels(init_cfg, submodels, True)

    sys.modules['transformers'] = transformer_location


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
