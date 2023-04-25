# Copyright (c) OpenMMLab. All rights reserved.
from mmagic.models.archs import conv


def test_conv():
    assert 'Deconv' in conv.MODELS.module_dict
