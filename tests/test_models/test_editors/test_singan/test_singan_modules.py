# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmagic.models.editors.singan.singan_modules import (DiscriminatorBlock,
                                                         GeneratorBlock)


def test_GeneratorBlock():
    gen_block = GeneratorBlock(3, 6, 3, 1, 3, 4, 4)
    x = torch.randn(1, 3, 6, 6)
    prev = torch.randn(1, 6, 6, 6)
    out = gen_block(x, prev)
    assert out.shape == (1, 6, 6, 6)

    gen_block = GeneratorBlock(3, 6, 3, 1, 3, 4, 4, allow_no_residual=True)
    x = torch.randn(1, 3, 6, 6)
    prev = torch.randn(1, 3, 6, 6)
    out = gen_block(x, prev)
    assert out.shape == (1, 6, 6, 6)


def test_DiscriminatorBlock():
    disc_block = DiscriminatorBlock(3, 4, 1, 3, 1, 3)
    x = torch.randn(1, 3, 6, 6)
    out = disc_block(x)
    assert out.shape == (1, 1, 6, 6)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
