# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmagic.models.editors.inst_colorization.weight_layer import WeightLayer


@pytest.mark.skipif(
    'win' in platform.system().lower() and 'cu' in torch.__version__,
    reason='skip on windows-cuda due to limited RAM.')
def test_weight_layer():

    weight_layer = WeightLayer(64)

    instance_feature_conv1_2 = torch.rand(1, 64, 256, 256)
    conv1_2 = torch.rand(1, 64, 256, 256)
    box_info = torch.tensor([[175, 29, 96, 54, 52, 106],
                             [14, 191, 84, 61, 51, 111],
                             [117, 64, 115, 46, 75, 95],
                             [41, 165, 121, 47, 50, 88],
                             [46, 136, 94, 45, 74, 117],
                             [79, 124, 62, 115, 53, 79],
                             [156, 64, 77, 138, 36, 41],
                             [200, 48, 114, 131, 8, 11],
                             [115, 78, 92, 81, 63, 83]])
    conv1_2 = weight_layer(instance_feature_conv1_2, conv1_2, box_info)

    assert conv1_2.shape == instance_feature_conv1_2.shape


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
