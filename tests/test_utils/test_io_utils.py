# Copyright (c) OpenMMLab. All rights reserved.

from mmagic.utils.io_utils import download_from_url


def test_download_from_url():
    # test to download a small file
    dest_path = download_from_url(
        'https://download.openmmlab.com/mmgen/dataset/singan/balloons.png',
        dest_path='./')
    print(dest_path)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
