# Copyright (c) OpenMMLab. All rights reserved.
from mmagic.datasets.transforms import blur_kernels


def test_blur_kernels():
    kernels = [
        'iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso',
        'plateau_aniso', 'sinc'
    ]
    for kernel_type in kernels:
        kernel = blur_kernels.random_mixed_kernels([kernel_type], [1], 5)
        assert kernel.shape == (5, 5)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
