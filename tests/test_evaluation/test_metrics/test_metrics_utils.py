# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

from mmedit.evaluation.metrics import metrics_utils


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
    print(img)
    new_img = metrics_utils.img_transform(img, 0, 'HWC', None, 'rgb')
    assert new_img.shape == (4, 4, 3)


def test_obtain_data():
    img = np.random.randint(0, 255, size=(4, 4, 3))
    key = 'img'
    data_sample = {'data_samples': {key: img}}
    result = metrics_utils.obtain_data(data_sample, key)
    assert not (result - img).any()


def test_to_numpy():
    input = torch.rand(1, 3, 8, 8)
    output = metrics_utils.to_numpy(input)
    assert isinstance(output, np.ndarray)
