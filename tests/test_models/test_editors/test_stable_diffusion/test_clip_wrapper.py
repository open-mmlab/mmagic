# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from transformers import CLIPConfig

from mmedit.models.editors.stable_diffusion.clip_wrapper import (
    StableDiffusionSafetyChecker, load_clip_submodels)


def test_clip_wrapper():
    clipconfig = CLIPConfig()
    safety_checker = StableDiffusionSafetyChecker(clipconfig)

    clip_input = torch.rand((1, 3, 224, 224))
    images_input = torch.rand((1, 512, 512, 3))

    result = safety_checker.forward(clip_input, images_input)
    assert result[0].shape == (1, 512, 512, 3)


def test_load_clip_submodels():
    init_cfg = dict(
        type='Pretrained',
        pretrained_model_path='tem',
    )

    submodels = []
    with pytest.raises(Exception):
        load_clip_submodels(init_cfg, submodels, True)
