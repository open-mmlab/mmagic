# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmagic.models.utils.sampling_utils import label_sample_fn, noise_sample_fn


def test_noise_sample_fn():
    # test noise is a callable function
    noise_callable = torch.randn
    noise = noise_sample_fn(noise_callable, noise_size=(2, 3), device='cpu')
    assert noise.shape == (1, 2, 3)


def test_label_sample_fn():
    label = label_sample_fn(None, num_classes=-1)
    assert label is None
    label = label_sample_fn(None)
    assert label is None

    label_inp = torch.randint(0, 10, (1, ))
    assert (label_sample_fn(label_inp, num_classes=10) == label_inp).all()

    # np.ndarray input
    label_inp = np.array([3, 2, 1])
    tar_label = torch.LongTensor([3, 2, 1])
    assert (label_sample_fn(label_inp, num_classes=10,
                            device='cpu') == tar_label).all()

    # list input
    label_inp = [0, 1, 1]
    tar_label = torch.LongTensor([0, 1, 1])
    assert (label_sample_fn(label_inp, num_classes=10) == tar_label).all()
    label_inp = [np.array([0]), np.array([1]), np.array([0])]
    tar_label = torch.LongTensor([0, 1, 0])
    assert (label_sample_fn(label_inp, num_classes=10) == tar_label).all()

    label_inp = [
        torch.LongTensor([0]),
        torch.LongTensor([1]),
        torch.LongTensor([0])
    ]
    tar_label = torch.LongTensor([0, 1, 0])
    assert (label_sample_fn(label_inp, num_classes=10) == tar_label).all()

    # list input --> raise error
    label_inp = ['1', '2']
    with pytest.raises(AssertionError):
        label_sample_fn(label_inp, num_classes=10)

    # callable input
    def label_function(num_batches):
        return torch.randint(0, 3, size=(num_batches, ))

    assert label_sample_fn(label_function, num_batches=3).shape == (3, )

    # test raise error
    with pytest.raises(AssertionError):
        label_sample_fn(torch.randn(3, 3), num_classes=10)

    with pytest.raises(AssertionError):
        label_sample_fn(torch.randint(0, 3, (2, 2)))

    with pytest.raises(AssertionError):
        label_sample_fn([0, 10, 2, 3], num_classes=5)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
