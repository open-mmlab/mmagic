# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmedit.models.inpaintors import AOTBlockNeck


def test_aot_dilation_neck():
    neck = AOTBlockNeck(
        in_channels=256, dilation_rates=(1, 2, 4, 8), num_aotblock=8)
    x = torch.rand((2, 256, 64, 64))
    res = neck(x)
    assert res.shape == (2, 256, 64, 64)

    if torch.cuda.is_available():
        neck = AOTBlockNeck(
            in_channels=256, dilation_rates=(1, 2, 4, 8),
            num_aotblock=8).cuda()
        x = torch.rand((2, 256, 64, 64)).cuda()
        res = neck(x)
        assert res.shape == (2, 256, 64, 64)
