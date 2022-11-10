# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmedit.models import AirNetRestorer
from mmedit.utils import register_all_modules

register_all_modules()


def test_airnetrestorer():

    model = AirNetRestorer(
        generator=dict(
            type='AirNet',
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
            )),
        pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
        train_cfg=dict(),
        test_cfg=dict(),
        data_preprocessor=dict(
            type='EditDataPreprocessor',
            mean=[0., 0., 0.],
            std=[255., 255., 255.],
        ),
        train_patch_size=128)

    # test attributes
    assert model.__class__.__name__ == 'AirNetRestorer'

    # prepare data
    inputs = torch.rand(1, 3, 128, 128)
    targets = torch.rand(1, 3, 128, 128)

    # test on cpu
    output = model.forward_tensor(inputs)
    assert torch.is_tensor(output)
    assert output.shape == targets.shape

    # test on gpu
    if torch.cuda.is_available():
        model = model.cuda()
        inputs = inputs.cuda()
        targets = targets.cuda()

        output = model.forward_tensor(inputs)
        assert torch.is_tensor(output)
        assert output.shape == targets.shape
