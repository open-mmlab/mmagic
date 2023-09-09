# Copyright (c) OpenMMLab. All rights reserved.
from mmagic.utils import print_colored_log


def test_print_colored_log():
    print_colored_log('Test print_colored_log info.')


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
