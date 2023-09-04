# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmagic.models.editors import AOTBlockNeck


def test_aot_dilation_neck():
    neck = AOTBlockNeck(
        in_channels=32, dilation_rates=(1, 2, 4, 8), num_aotblock=4)
    x = torch.rand((2, 32, 64, 64))
    res = neck(x)
    assert res.shape == (2, 32, 64, 64)

    if torch.cuda.is_available():
        neck = AOTBlockNeck(
            in_channels=32, dilation_rates=(1, 2, 4, 8),
            num_aotblock=4).cuda()
        x = torch.rand((2, 32, 64, 64)).cuda()
        res = neck(x)
        assert res.shape == (2, 32, 64, 64)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
