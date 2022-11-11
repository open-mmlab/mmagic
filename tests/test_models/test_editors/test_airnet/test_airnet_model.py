# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmedit.models import AirNetRestorer
from mmedit.structures import EditDataSample, PixelData
from mmedit.utils import register_all_modules

register_all_modules()


@pytest.mark.skipif(
    'win' in platform.system().lower() and 'cu' in torch.__version__,
    reason='skip on windows-cuda due to limited RAM.')
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
        train_cfg=dict(epochs_encoder=0),
        test_cfg=dict(),
        data_preprocessor=dict(
            type='EditDataPreprocessor',
            mean=[0., 0., 0.],
            std=[255., 255., 255.],
        ),
        train_patch_size=64)

    # test attributes
    assert model.__class__.__name__ == 'AirNetRestorer'

    # prepare data
    inputs = torch.rand(1, 3, 128, 128)
    targets = torch.rand(1, 3, 128, 128)
    data_sample = [
        EditDataSample(
            metainfo=dict(coord=torch.rand(256, 2), cell=torch.rand(256, 2)))
    ]
    data_sample[0].gt_img = PixelData(data=targets.squeeze())

    # test on cpu
    output = model.forward_tensor(inputs)
    assert torch.is_tensor(output)
    assert output.shape == targets.shape

    out_dict = model._double_crop(inputs, data_sample)
    assert out_dict['degrad_patch_1'].shape == torch.Size([1, 3, 64, 64])
    assert out_dict['degrad_patch_2'].shape == torch.Size([1, 3, 64, 64])
    assert out_dict['clear_patch_1'].shape == torch.Size([1, 3, 64, 64])
    assert out_dict['clear_patch_2'].shape == torch.Size([1, 3, 64, 64])

    patch1, patch2 = model._crop_patch(inputs, targets)
    assert torch.is_tensor(patch1)
    assert torch.is_tensor(patch2)
    assert patch1.shape == patch2.shape == torch.Size([1, 3, 64, 64])

    # test on gpu
    if torch.cuda.is_available():
        model = model.cuda()
        inputs = inputs.cuda()
        targets = targets.cuda()

        output = model.forward_tensor(inputs)
        assert torch.is_tensor(output)
        assert output.shape == targets.shape

        out_dict = model._double_crop(inputs, data_sample)
        assert out_dict['degrad_patch_1'].shape == torch.Size([1, 3, 64, 64])
        assert out_dict['degrad_patch_2'].shape == torch.Size([1, 3, 64, 64])
        assert out_dict['clear_patch_1'].shape == torch.Size([1, 3, 64, 64])
        assert out_dict['clear_patch_2'].shape == torch.Size([1, 3, 64, 64])

        patch1, patch2 = model._crop_patch(inputs, targets)
        assert torch.is_tensor(patch1)
        assert torch.is_tensor(patch2)
        assert patch1.shape == patch2.shape == torch.Size([1, 3, 64, 64])
