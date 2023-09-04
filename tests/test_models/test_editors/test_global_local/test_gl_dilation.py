# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmagic.models.archs import SimpleGatedConvModule
from mmagic.models.editors.global_local import GLDilationNeck
from mmagic.registry import MODELS
from mmagic.utils import register_all_modules

register_all_modules()


def test_gl_dilation_neck():
    x = torch.rand((2, 8, 64, 64))
    template_cfg = dict(type='GLDilationNeck', in_channels=8)

    neck = MODELS.build(template_cfg)
    res = neck(x)
    assert res.shape == (2, 8, 64, 64)

    if torch.cuda.is_available():
        neck = GLDilationNeck(in_channels=8).cuda()
        x = torch.rand((2, 8, 64, 64)).cuda()
        res = neck(x)
        assert res.shape == (2, 8, 64, 64)

        neck = GLDilationNeck(in_channels=8, conv_type='gated_conv').cuda()
        res = neck(x)
        assert isinstance(neck.dilation_convs[0], SimpleGatedConvModule)
        assert res.shape == (2, 8, 64, 64)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
