# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmagic.models.archs import pixel_unshuffle


def test_pixel_unshuffle():
    # test on cpu
    x = torch.rand(1, 3, 20, 20)
    y = pixel_unshuffle(x, scale=2)
    assert y.shape == (1, 12, 10, 10)
    with pytest.raises(AssertionError):
        y = pixel_unshuffle(x, scale=3)

    # test on gpu
    if torch.cuda.is_available():
        x = x.cuda()
        y = pixel_unshuffle(x, scale=2)
        assert y.shape == (1, 12, 10, 10)

        with pytest.raises(AssertionError):
            y = pixel_unshuffle(x, scale=3)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
