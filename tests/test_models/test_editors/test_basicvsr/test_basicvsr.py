# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from mmengine.optim import OptimWrapper
from torch.optim import Adam

from mmagic.models import BasicVSR, BasicVSRNet, DataPreprocessor
from mmagic.models.losses import CharbonnierLoss
from mmagic.structures import DataSample


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
        data_preprocessor=DataPreprocessor())

    assert isinstance(model, BasicVSR)
    assert isinstance(model.generator, BasicVSRNet)
    assert isinstance(model.pixel_loss, CharbonnierLoss)

    optimizer = Adam(model.generator.parameters(), lr=0.001)
    optim_wrapper = OptimWrapper(optimizer)

    # prepare data
    inputs = torch.rand(5, 3, 16, 16)
    target = torch.rand(5, 3, 64, 64)
    data_sample = DataSample(gt_img=target)
    data = dict(inputs=[inputs], data_samples=[data_sample])

    # train
    log_vars = model.train_step(data, optim_wrapper)
    assert model.generator.spynet.basic_module[0].basic_module[
        0].conv.weight.requires_grad is False
    assert isinstance(log_vars, dict)

    log_vars = model.train_step(data, optim_wrapper)
    assert model.generator.spynet.basic_module[0].basic_module[
        0].conv.weight.requires_grad is True
    assert isinstance(log_vars, dict)

    # val
    output = model.val_step(data)
    assert output[0].output.pred_img.shape == (5, 3, 64, 64)

    data['data_samples'][0].gt_img.data = torch.rand(3, 64, 64)
    output = model.val_step(data)
    assert output[0].output.pred_img.shape == (3, 64, 64)

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
        data_preprocessor=DataPreprocessor())

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
            data_preprocessor=DataPreprocessor())


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
