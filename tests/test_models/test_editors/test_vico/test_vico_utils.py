# Copyright (c) OpenMMLab. All rights reserved.

import platform

import pytest
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION

from mmagic.models.editors.vico.vico_utils import set_vico_modules


@pytest.mark.skipif(
    'win' in platform.system().lower()
    or digit_version(TORCH_VERSION) <= digit_version('1.8.1'),
    reason='skip on windows due to limited RAM'
    'and get_submodule requires torch >= 1.9.0')
def test_set_vico_modules():
    model = UNet2DConditionModel()
    image_cross_layers = [1] * 16
    set_vico_modules(model, image_cross_layers)

    # test set vico modules
    for name, layer in model.named_modules():
        if len(name.split('.')) > 1:
            module_name = name.split('.')[-2]
            if module_name == 'attentions':
                assert layer.__class__.__name__ == 'ViCoTransformer2D'
                assert hasattr(layer, 'image_cross_attention')
