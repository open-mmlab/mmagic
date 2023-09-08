# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmagic.models.editors.nafnet.naf_layerNorm2d import LayerNorm2d


def test_layer_norm():
    inputs = torch.ones((1, 3, 64, 64))
    targets = torch.zeros((1, 3, 64, 64))

    layer_norm_2d = LayerNorm2d(inputs.shape[1])
    outputs = layer_norm_2d(inputs)
    assert outputs.shape == targets.shape
    assert torch.all(torch.eq(outputs, targets))


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
