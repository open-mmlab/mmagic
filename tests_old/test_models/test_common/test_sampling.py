# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmedit.models.common import PixelShufflePack, pixel_unshuffle


def test_pixel_shuffle():

    # test on cpu
    model = PixelShufflePack(3, 3, 2, 3)
    model.init_weights()
    x = torch.rand(1, 3, 16, 16)
    y = model(x)
    assert y.shape == (1, 3, 32, 32)

    # test on gpu
    if torch.cuda.is_available():
        model = model.cuda()
        x = x.cuda()
        y = model(x)
        assert y.shape == (1, 3, 32, 32)


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
