# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmagic.models.editors.nafnet.naf_avgpool2d import NAFAvgPool2d


def test_avgpool2d():
    inputs = torch.ones((1, 3, 32, 32))
    targets = torch.Tensor([1., 1., 1.]).view(1, 3, 1, 1)
    tar_info = 'kernel_size=None, base_size=(48, 48),' \
        + ' stride=None, fast_imp=False'
    base_size = (int(inputs.shape[-2] * 1.5), int(inputs.shape[-1] * 1.5))
    train_size = inputs.shape

    avg_pool_2d = NAFAvgPool2d(
        base_size=base_size, train_size=train_size, fast_imp=False)
    info = avg_pool_2d.extra_repr()
    outputs = avg_pool_2d(inputs)
    print(outputs)
    assert info == tar_info
    assert torch.all(torch.eq(targets, outputs))

    avg_pool_2d = NAFAvgPool2d(
        base_size=int(inputs.shape[-2] * 1.5),
        train_size=train_size,
        fast_imp=False)
    info = avg_pool_2d.extra_repr()
    tar_info = 'kernel_size=None, base_size=48,' \
        + ' stride=None, fast_imp=False'
    outputs = avg_pool_2d(inputs)
    print(outputs)
    assert info == tar_info
    assert torch.all(torch.eq(targets, outputs))

    avg_pool_2d = NAFAvgPool2d(
        base_size=base_size, train_size=train_size, fast_imp=True)
    info = avg_pool_2d.extra_repr()
    tar_info = 'kernel_size=None, base_size=(48, 48),' \
        + ' stride=None, fast_imp=True'
    outputs = avg_pool_2d(inputs)
    print(outputs)
    assert info == tar_info
    assert torch.all(torch.eq(targets, outputs))

    avg_pool_2d = NAFAvgPool2d(
        base_size=(16, 16), train_size=train_size, fast_imp=True)
    info = avg_pool_2d.extra_repr()
    tar_info = 'kernel_size=None, base_size=(16, 16),' \
        + ' stride=None, fast_imp=True'
    outputs = avg_pool_2d(inputs)
    print(outputs)
    assert info == tar_info
    assert torch.all(torch.eq(targets, outputs))


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
