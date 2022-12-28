# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmedit.models.editors import IFRNet
from mmedit.structures import EditDataSample, PixelData
from mmedit.utils import register_all_modules

register_all_modules()


def test_ifrnet():

    model1 = IFRNet(
        generator=dict(type='IFRNetInterpolator'),
        pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
        interpolation_scale=2,
        data_preprocessor=dict(
            type='EditDataPreprocessor',
            mean=[0.0, 0.0, 0.0],
            std=[255.0, 255.0, 255.0]))

    model2 = IFRNet(
        generator=dict(type='IFRNetInterpolator'),
        pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
        interpolation_scale=8,
        data_preprocessor=dict(
            type='EditDataPreprocessor',
            mean=[0.0, 0.0, 0.0],
            std=[255.0, 255.0, 255.0]))

    # prepare inputs
    inputs = torch.ones(1, 2, 3, 64, 64)
    data_sample = [
        EditDataSample(_gt_img=PixelData(data=torch.ones(1, 3, 64, 64)))
    ]
    model1.data_preprocessor(dict(inputs=inputs, data_samples=data_sample))
    model2.data_preprocessor(dict(inputs=inputs, data_samples=data_sample))

    # forward tensor
    pred = model1.forward_tensor(inputs)
    assert pred.shape == torch.Size([1, 1, 3, 64, 64])
    pred = model2.forward_tensor(inputs)
    assert pred.shape == torch.Size([1, 7, 3, 64, 64])

    # forward inference
    infer = model1.forward_inference(inputs, data_sample)
    assert len(infer) == 1
    assert infer[0].pred_img.data.shape == torch.Size([3, 64, 64])
    infer = model2.forward_inference(inputs, data_sample)
    assert len(infer) == 1
    assert infer[0].pred_img.data.shape == torch.Size([7, 3, 64, 64])

    # forward train
    loss = model1.forward_train(inputs, data_sample)
    assert isinstance(loss, dict)
    assert 'loss_rec' in loss
    assert 'loss_geo' in loss
