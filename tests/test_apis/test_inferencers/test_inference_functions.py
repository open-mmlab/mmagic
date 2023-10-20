# Copyright (c) OpenMMLab. All rights reserved.
import pytest

from mmagic.apis.inferencers.inference_functions import (calculate_grid_size,
                                                         init_model,
                                                         set_random_seed)


def test_init_model():
    set_random_seed(1)

    with pytest.raises(TypeError):
        init_model(['dog'])


def test_calculate_grid_size():
    inp_batch_size = (10, 13, 20, 1, 4)
    target_nrow = (4, 4, 5, 1, 2)
    for bz, tar in zip(inp_batch_size, target_nrow):
        assert calculate_grid_size(bz) == tar

    # test aspect_ratio is not None
    inp_batch_size = (10, 13, 20, 1, 4)
    aspect_ratio = (2, 3, 3, 4, 3)
    target_nrow = (3, 3, 3, 1, 2)
    for bz, ratio, tar in zip(inp_batch_size, aspect_ratio, target_nrow):
        assert calculate_grid_size(bz, ratio) == tar


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
