# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmagic.datasets.transforms import RandomDownSampling


def test_random_down_sampling():
    img1 = np.uint8(np.random.randn(480, 640, 3) * 255)
    inputs1 = dict(gt=img1)
    down_sampling1 = RandomDownSampling(
        scale_min=1, scale_max=4, patch_size=None)
    results1 = down_sampling1(inputs1)
    assert set(list(results1.keys())) == set(['gt', 'img', 'scale'])
    assert repr(down_sampling1) == (
        down_sampling1.__class__.__name__ +
        f' scale_min={down_sampling1.scale_min}, ' +
        f'scale_max={down_sampling1.scale_max}, ' +
        f'patch_size={down_sampling1.patch_size}, ' +
        f'interpolation={down_sampling1.interpolation}, ' +
        f'backend={down_sampling1.backend}')

    img2 = np.uint8(np.random.randn(480, 640, 3) * 255)
    inputs2 = dict(gt=img2)
    down_sampling2 = RandomDownSampling(
        scale_min=1, scale_max=4, patch_size=48)
    results2 = down_sampling2(inputs2)
    assert set(list(results2.keys())) == set(['gt', 'img', 'scale'])
    assert repr(down_sampling2) == (
        down_sampling2.__class__.__name__ +
        f' scale_min={down_sampling2.scale_min}, ' +
        f'scale_max={down_sampling2.scale_max}, ' +
        f'patch_size={down_sampling2.patch_size}, ' +
        f'interpolation={down_sampling2.interpolation}, ' +
        f'backend={down_sampling2.backend}')

    # test copy meta info
    img3 = np.uint8(np.random.randn(480, 640, 3) * 255)
    inputs3 = dict(gt=img3, gt_channel_order='rgb', gt_color_type='color')
    down_sampling3 = RandomDownSampling(
        scale_min=1, scale_max=4, patch_size=48)
    results3 = down_sampling3(inputs3)
    assert results3['img_channel_order'] == 'rgb'
    assert results3['img_color_type'] == 'color'


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
