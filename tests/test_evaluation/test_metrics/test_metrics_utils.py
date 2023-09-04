# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest

from mmagic.evaluation.metrics import metrics_utils
from mmagic.evaluation.metrics.metrics_utils import reorder_image


def test_average():
    results = []
    key = 'data'
    total = 0
    n = 0
    for i in range(10):
        results.append({'batch_size': i, key: i})
        total += i * i
        n += i
    average = metrics_utils.average(results, key)
    assert average == total / n


def test_img_transform():
    img = np.random.randint(0, 255, size=(4, 4, 3))
    new_img = metrics_utils.img_transform(img, 0, 'HWC', None, 'rgb')
    assert new_img.shape == (4, 4, 3)


def test_obtain_data():
    img = np.random.randint(0, 255, size=(4, 4, 3))
    key = 'img'
    data_sample = {'data_samples': {key: img}}
    result = metrics_utils.obtain_data(data_sample, key)
    assert not (result - img).any()


def test_reorder_image():
    img_hw = np.ones((32, 32))
    img_hwc = np.ones((32, 32, 3))
    img_chw = np.ones((3, 32, 32))

    with pytest.raises(ValueError):
        reorder_image(img_hw, 'HH')

    output = reorder_image(img_hw)
    assert output.shape == (32, 32, 1)

    output = reorder_image(img_hwc)
    assert output.shape == (32, 32, 3)

    output = reorder_image(img_chw, input_order='CHW')
    assert output.shape == (32, 32, 3)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
