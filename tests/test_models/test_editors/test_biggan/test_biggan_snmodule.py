# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmagic.models.editors.biggan.biggan_snmodule import SpectralNorm


class MyBlock(nn.Module):

    def __init__(self, num_channels):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_channels))

    def forward(self, x):
        return x + self.weight


class MySNBlock(MyBlock, SpectralNorm):

    def __init__(self, num_channels):
        MyBlock.__init__(self, num_channels)
        SpectralNorm.__init__(
            self, num_svs=2, num_iters=2, num_outputs=num_channels)

    def forward(self, x):
        return x + self.sn_weight()


# test Spectral Norm with my own layer
def test_SpectralNorm():
    sn_block = MySNBlock(num_channels=4)
    x = torch.randn(1, 4)
    out = sn_block(x)
    assert out.shape == (1, 4)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
