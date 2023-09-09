# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmagic.models.editors.wgan_gp.wgan_gp_module import (ConvLNModule,
                                                          WGANDecisionHead,
                                                          WGANNoiseTo2DFeat)


def test_ConvLNModule():
    # test norm_cfg is None
    conv = ConvLNModule(3, 6, 3, 1, 1)
    assert conv.norm is None


def test_WGANDecisionHead():
    head = WGANDecisionHead(3, 8, 4)
    x = torch.randn(1, 3, 4, 4)
    out = head(x)
    assert out.shape == (1, 4)


def test_WGANNoiseTo2DFeat():
    noise2feat = WGANNoiseTo2DFeat(16, 32)
    noise = torch.randn(1, 16)
    feat = noise2feat(noise)
    assert feat.shape == (1, 32, 4, 4)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
