# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.optim import OptimWrapper
from torch.optim import Adam

from mmedit.data_element import EditDataSample, PixelData
from mmedit.models.data_processor import EditDataPreprocessor
from mmedit.models.losses import MSELoss
from mmedit.models.video_restorers import TDAN, TDANNet


def test_tdan():

    model = TDAN(
        generator=dict(type='TDANNet'),
        pixel_loss=dict(type='MSELoss', loss_weight=1.0, reduction='mean'),
        lq_pixel_loss=dict(type='MSELoss', loss_weight=0.01, reduction='mean'),
        data_preprocessor=EditDataPreprocessor(mean=[0.5, 0.5, 0.5]))

    assert model.__class__.__name__ == 'TDAN'
    assert isinstance(model.generator, TDANNet)
    assert isinstance(model.pixel_loss, MSELoss)
    assert isinstance(model.data_preprocessor, EditDataPreprocessor)

    optimizer = Adam(model.generator.parameters(), lr=0.001)
    optim_wrapper = OptimWrapper(optimizer)

    # prepare data
    inputs = torch.rand(5, 3, 16, 16)
    target = torch.rand(3, 64, 64)
    data_sample = EditDataSample(gt_img=PixelData(data=target))
    data = [dict(inputs=inputs, data_sample=data_sample)]

    # train
    log_vars = model.train_step(data, optim_wrapper)
    assert isinstance(log_vars, dict)

    # val
    output = model.val_step(data)
    assert output[0].pred_img.data.shape == (3, 64, 64)

    # feat
    output = model(torch.rand(1, 5, 3, 16, 16), mode='tensor')
    assert output.shape == (1, 3, 64, 64)
