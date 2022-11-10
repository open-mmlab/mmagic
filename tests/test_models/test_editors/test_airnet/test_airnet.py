# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmedit.models import AirNet


def test_airnet():
    model = AirNet(
        encoder_cfg=dict(
            type='CBDE',
            batch_size=1,
            dim=256,
        ),
        restorer_cfg=dict(
            type='DGRN',
            n_groups=5,
            n_blocks=5,
            n_feats=64,
            kernel_size=3,
        ),
    )

    # test attributes
    assert model.__class__.__name__ == 'AirNet'

    # prepare data
    inputs = torch.rand(1, 3, 64, 64)
    targets = torch.rand(1, 3, 64, 64)

    # test on cpu
    output = model(inputs)['restored']
    assert torch.is_tensor(output)
    assert output.shape == targets.shape

    # test on gpu
    if torch.cuda.is_available():
        model = model.cuda()
        inputs = inputs.cuda()
        targets = targets.cuda()

        output = model(inputs)['restored']
        assert torch.is_tensor(output)
        assert output.shape == targets.shape
