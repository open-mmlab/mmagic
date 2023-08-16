# Copyright (c) OpenMMLab. All rights reserved.

import pytest
import torch
import torch.nn as nn
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION

from mmagic.models.editors.vico.vico_utils import set_vico_modules
from diffusers.models.unet_2d_condition import UNet2DConditionModel


@pytest.mark.skipif(
    digit_version(TORCH_VERSION) <= digit_version('1.8.1'),
    reason='get_submodule requires torch >= 1.9.0')
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

    model.train()
    img = torch.randn(1, 4, 64, 64)
    img_ref = torch.randn(1, 4, 64, 64)
    img = torch.cat([img, img_ref])
    context = torch.randn(1, 77, 768)
    timesteps = torch.LongTensor(981)
    ph_pos = (torch.IntTensor(1), torch.IntTensor(1))

    # out = model(img, timesteps, context, placeholder_position=ph_pos)
    # assert len(out) == 2
    # assert 'sample' in out and 'loss_reg' in out
    # assert out['sample'].shape == (1, 4, 64, 64)

if __name__ == "__main__":
    model = UNet2DConditionModel()
    for name, layer in model.named_modules():
        print(name, '->', layer.__class__.__name__)
