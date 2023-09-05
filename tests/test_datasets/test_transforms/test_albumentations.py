# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmagic.datasets.transforms import Albumentations


def test_albumentations():
    results = {}
    results['img'] = np.ones((8, 8, 3)).astype(np.uint8)
    model = Albumentations(
        keys=['img'], transforms=[
            dict(type='Resize', height=4, width=4),
        ])
    results = model(results)
    assert results['img'].shape == (4, 4, 3)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
