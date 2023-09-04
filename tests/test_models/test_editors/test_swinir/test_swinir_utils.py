# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmagic.models.editors.swinir.swinir_utils import (drop_path, to_2tuple,
                                                       window_partition,
                                                       window_reverse)


def test_drop_path():
    x = torch.randn(1, 3, 8, 8)
    x = drop_path(x)
    assert x.shape == (1, 3, 8, 8)


def test_to_2tuple():
    x = 8
    x = to_2tuple(x)
    assert x == (8, 8)


def test_window():
    x = torch.randn(1, 8, 8, 3)
    x = window_partition(x, 4)
    assert x.shape == (4, 4, 4, 3)
    x = window_reverse(x, 4, 8, 8)
    assert x.shape == (1, 8, 8, 3)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
