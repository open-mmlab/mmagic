import numpy as np

from mmedit.datasets.pipelines import DownSampling


def test_down_sampling():
    img1 = np.uint8(np.random.randn(480, 640, 3) * 255)
    inputs1 = dict(gt=img1)
    down_sampling1 = DownSampling(scale_min=1, scale_max=4, patch_size=None)
    results1 = down_sampling1(inputs1)
    assert set(list(results1.keys())) == set(['gt', 'lq', 'scale'])
    assert repr(down_sampling1) == (
        down_sampling1.__class__.__name__ +
        f'scale_min={down_sampling1.scale_min}, ' +
        f'scale_max={down_sampling1.scale_max}, ' +
        f'patch_size={down_sampling1.patch_size}')

    img2 = np.uint8(np.random.randn(480, 640, 3) * 255)
    inputs2 = dict(gt=img2)
    down_sampling2 = DownSampling(scale_min=1, scale_max=4, patch_size=48)
    results2 = down_sampling2(inputs2)
    assert set(list(results2.keys())) == set(['gt', 'lq', 'scale'])
    assert repr(down_sampling2) == (
        down_sampling2.__class__.__name__ +
        f'scale_min={down_sampling2.scale_min}, ' +
        f'scale_max={down_sampling2.scale_max}, ' +
        f'patch_size={down_sampling2.patch_size}')
