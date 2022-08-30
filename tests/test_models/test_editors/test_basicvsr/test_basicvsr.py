# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from mmengine.optim import OptimWrapper
from torch.optim import Adam

from mmedit.models import BasicVSR, BasicVSRNet, EditDataPreprocessor
from mmedit.models.losses import CharbonnierLoss
from mmedit.structures import EditDataSample, PixelData


def test_basicvsr():

    model = BasicVSR(
        generator=dict(
            type='BasicVSRNet',
            mid_channels=8,
            num_blocks=1,
            spynet_pretrained=None),
        pixel_loss=dict(
            type='CharbonnierLoss', loss_weight=1.0, reduction='mean'),
        train_cfg=dict(fix_iter=1),
        ensemble=None,
        data_preprocessor=EditDataPreprocessor())

    assert isinstance(model, BasicVSR)
    assert isinstance(model.generator, BasicVSRNet)
    assert isinstance(model.pixel_loss, CharbonnierLoss)

    optimizer = Adam(model.generator.parameters(), lr=0.001)
    optim_wrapper = OptimWrapper(optimizer)

    # prepare data
    inputs = torch.rand(5, 3, 64, 64)
    target = torch.rand(5, 3, 256, 256)
    data_sample = EditDataSample(gt_img=PixelData(data=target))
    data = [dict(inputs=inputs, data_sample=data_sample)]

    # train
    log_vars = model.train_step(data, optim_wrapper)
    # print(model.generator.spynet.basic_module[0].basic_module[0].conv.weight)
    assert model.generator.spynet.basic_module[0].basic_module[
        0].conv.weight.requires_grad is False
    assert isinstance(log_vars, dict)
    log_vars = model.train_step(data, optim_wrapper)
    assert model.generator.spynet.basic_module[0].basic_module[
        0].conv.weight.requires_grad is True
    assert isinstance(log_vars, dict)

    # val
    output = model.val_step(data)
    assert output[0].pred_img.data.shape == (5, 3, 256, 256)
    data[0]['data_samples'].gt_img.data = torch.rand(3, 256, 256)
    output = model.val_step(data)
    assert output[0].pred_img.data.shape == (3, 256, 256)
    img = torch.rand(3, 64, 64)
    data[0]['inputs'] = torch.stack([img, img])
    output = model.val_step(data)
    assert output[0].pred_img.data.shape == (3, 256, 256)

    model = BasicVSR(
        generator=dict(
            type='BasicVSRNet',
            mid_channels=8,
            num_blocks=1,
            spynet_pretrained=None),
        pixel_loss=dict(
            type='CharbonnierLoss', loss_weight=1.0, reduction='mean'),
        train_cfg=dict(fix_iter=1),
        ensemble=dict(type='SpatialTemporalEnsemble'),
        data_preprocessor=EditDataPreprocessor())

    with pytest.raises(NotImplementedError):
        model = BasicVSR(
            generator=dict(
                type='BasicVSRNet',
                mid_channels=8,
                num_blocks=1,
                spynet_pretrained=None),
            pixel_loss=dict(
                type='CharbonnierLoss', loss_weight=1.0, reduction='mean'),
            train_cfg=dict(fix_iter=1),
            ensemble=dict(type=''),
            data_preprocessor=EditDataPreprocessor())
