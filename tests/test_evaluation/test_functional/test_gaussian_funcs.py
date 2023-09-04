# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmagic.evaluation.functional import gauss_gradient


def test_gauss_gradient():
    img = np.random.randint(0, 255, size=(8, 8, 3))
    grad = gauss_gradient(img, 1.4)
    assert grad.shape == (8, 8, 3)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
