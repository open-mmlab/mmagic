# Copyright (c) OpenMMLab. All rights reserved.
from mmedit.models.layers import conv


def test_conv():
    assert 'Deconv' in conv.CONV_LAYERS.module_dict
