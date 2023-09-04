# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmagic.models import NAFNet, NAFNetLocal


def test_nafnet():

    model = NAFNet(
        img_channels=3,
        mid_channels=64,
        enc_blk_nums=[2, 2, 4, 8],
        middle_blk_num=12,
        dec_blk_nums=[2, 2, 2, 2],
    )

    # test attributes
    assert model.__class__.__name__ == 'NAFNet'

    # prepare data
    inputs = torch.rand(1, 3, 64, 64)
    targets = torch.rand(1, 3, 64, 64)

    # test on cpu
    output = model(inputs)
    assert torch.is_tensor(output)
    assert output.shape == targets.shape

    # test on gpu
    if torch.cuda.is_available():
        model = model.cuda()
        inputs = inputs.cuda()
        targets = targets.cuda()
        output = model(inputs)
        assert torch.is_tensor(output)
        assert output.shape == targets.shape


def test_nafnet_local():

    model = NAFNetLocal(
        img_channels=3,
        mid_channels=64,
        enc_blk_nums=[1, 1, 1, 28],
        middle_blk_num=1,
        dec_blk_nums=[1, 1, 1, 1],
    )

    # test attributes
    assert model.__class__.__name__ == 'NAFNetLocal'

    # prepare data
    inputs = torch.rand(1, 3, 64, 64)
    targets = torch.rand(1, 3, 64, 64)

    # test on cpu
    output = model(inputs)
    assert torch.is_tensor(output)
    assert output.shape == targets.shape

    # test on gpu
    if torch.cuda.is_available():
        model = model.cuda()
        inputs = inputs.cuda()
        targets = targets.cuda()
        output = model(inputs)
        assert torch.is_tensor(output)
        assert output.shape == targets.shape


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
