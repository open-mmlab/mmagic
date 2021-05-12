import numpy as np

from mmedit.datasets.pipelines import RandomDownSampling, SRResize


def test_random_down_sampling():
    img1 = np.uint8(np.random.randn(480, 640, 3) * 255)
    inputs1 = dict(gt=img1)
    down_sampling1 = RandomDownSampling(
        scale_min=1, scale_max=4, patch_size=None)
    results1 = down_sampling1(inputs1)
    assert set(list(results1.keys())) == set(['gt', 'lq', 'scale'])
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
    assert set(list(results2.keys())) == set(['gt', 'lq', 'scale'])
    assert repr(down_sampling2) == (
        down_sampling2.__class__.__name__ +
        f' scale_min={down_sampling2.scale_min}, ' +
        f'scale_max={down_sampling2.scale_max}, ' +
        f'patch_size={down_sampling2.patch_size}, ' +
        f'interpolation={down_sampling2.interpolation}, ' +
        f'backend={down_sampling2.backend}')


def test_sr_resize():
    img = np.uint8(np.random.randn(480, 640, 3) * 255)
    inputs = dict(gt=img)
    re_size = SRResize(scale=1 / 4, input_key='gt', output_key='lq')
    results = re_size(inputs)
    assert set(list(results.keys())) == set(['gt', 'lq'])
    assert results['lq'].shape == (120, 160, 3)
    assert repr(re_size) == (
        re_size.__class__.__name__ + f' scale={re_size.scale}, ' +
        f'input_key={re_size.input_key}, ' +
        f'output_key={re_size.output_key}, ' +
        f'interpolation={re_size.interpolation}, ' +
        f'backend={re_size.backend}')

    inputs = dict(gt=img)
    re_size = SRResize(scale=2, input_key='gt', output_key='gt')
    results = re_size(inputs)
    assert set(list(results.keys())) == set(['gt'])
    assert results['gt'].shape == (960, 1280, 3)
