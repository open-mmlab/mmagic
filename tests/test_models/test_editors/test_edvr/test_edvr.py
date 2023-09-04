# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from mmengine.optim import OptimWrapper
from torch.optim import Adam

from mmagic.models import EDVR, EDVRNet
from mmagic.models.losses import CharbonnierLoss
from mmagic.structures import DataSample
from mmagic.utils import register_all_modules

register_all_modules()


def test_edvr():

    model = EDVR(
        generator=dict(
            type='EDVRNet',
            in_channels=3,
            out_channels=3,
            mid_channels=4,
            num_frames=5,
            deform_groups=1,
            num_blocks_extraction=1,
            num_blocks_reconstruction=1,
            center_frame_idx=2,
            with_tsa=True),
        pixel_loss=dict(
            type='CharbonnierLoss', loss_weight=1.0, reduction='sum'),
        train_cfg=dict(tsa_iter=5000),
        data_preprocessor=dict(type='DataPreprocessor'))

    assert isinstance(model, EDVR)
    assert isinstance(model.generator, EDVRNet)
    assert isinstance(model.pixel_loss, CharbonnierLoss)

    optimizer = Adam(model.generator.parameters(), lr=0.001)
    optim_wrapper = OptimWrapper(optimizer)

    # prepare data
    inputs = torch.rand(1, 5, 3, 20, 20)
    target = torch.rand(5, 3, 80, 80)
    data_sample = DataSample(gt_img=target)
    data = dict(inputs=inputs, data_samples=[data_sample])

    log_vars = model.train_step(data, optim_wrapper)
    assert model.generator.conv_first.weight.requires_grad is False
    assert isinstance(log_vars, dict)

    model.step_counter = torch.tensor(5000)
    log_vars = model.train_step(data, optim_wrapper)
    assert model.generator.conv_first.weight.requires_grad is True
    assert isinstance(log_vars, dict)

    with pytest.raises(KeyError):
        model = EDVR(
            generator=dict(
                type='EDVRNet',
                in_channels=3,
                out_channels=3,
                mid_channels=4,
                num_frames=5,
                deform_groups=1,
                num_blocks_extraction=1,
                num_blocks_reconstruction=1,
                center_frame_idx=2,
                with_tsa=True),
            pixel_loss=dict(
                type='CharbonnierLoss', loss_weight=1.0, reduction='sum'),
            train_cfg=dict())
        model.train_step(data, optim_wrapper)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
