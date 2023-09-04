# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmagic.models.editors import TOFlowVSRNet


def test_toflow_vsr_net():

    imgs = torch.rand(2, 7, 3, 16, 16)

    model = TOFlowVSRNet(adapt_official_weights=False)
    out = model(imgs)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 3, 16, 16)

    model = TOFlowVSRNet(adapt_official_weights=True)
    out = model(imgs)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 3, 16, 16)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
