# Copyright (c) OpenMMLab. All rights reserved.
from mmagic.utils import register_all_modules, try_import


def test_register_all_modules():
    register_all_modules()


def test_try_import():
    import numpy as np
    assert try_import('numpy') is np
    assert try_import('numpy111') is None


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
