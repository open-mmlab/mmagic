# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from mmengine.optim import OptimWrapper
from torch.optim import Adam

from mmedit.data_element import EditDataSample, PixelData
from mmedit.models.data_processor import EditDataPreprocessor
from mmedit.models.losses import L1Loss
from mmedit.models.video_interpolators import CAIN, CAINNet
from mmedit.registry import MODELS


def test_cain_net():

    model_cfg = dict(type='CAINNet')

    # build model
    model = MODELS.build(model_cfg)

    # test attributes
    assert model.__class__.__name__ == 'CAINNet'

    # prepare data
    inputs0 = torch.rand(1, 2, 3, 5, 5)
    target0 = torch.rand(1, 3, 5, 5)
    inputs = torch.rand(1, 2, 3, 256, 248)
    target = torch.rand(1, 3, 256, 248)

    # test on cpu
    output = model(inputs)
    output = model(inputs, padding_flag=True)
    model(inputs0, padding_flag=True)
    assert torch.is_tensor(output)
    assert output.shape == target.shape
    with pytest.raises(AssertionError):
        output = model(inputs[:, :1])

    model_cfg = dict(type='CAINNet', norm='in')
    model = MODELS.build(model_cfg)
    model(inputs)
    model_cfg = dict(type='CAINNet', norm='bn')
    model = MODELS.build(model_cfg)
    model(inputs)
    with pytest.raises(ValueError):
        model_cfg = dict(type='CAINNet', norm='lys')
        MODELS.build(model_cfg)

    # test on gpu
    if torch.cuda.is_available():
        model = model.cuda()
        inputs = inputs.cuda()
        target = target.cuda()
        output = model(inputs)
        output = model(inputs, True)
        assert torch.is_tensor(output)
        assert output.shape == target.shape
        inputs0 = inputs0.cuda()
        target0 = target0.cuda()
        model(inputs0, padding_flag=True)

        model_cfg = dict(type='CAINNet', norm='in')
        model = MODELS.build(model_cfg).cuda()
        model(inputs)
        model_cfg = dict(type='CAINNet', norm='bn')
        model = MODELS.build(model_cfg).cuda()
        model(inputs)
        with pytest.raises(ValueError):
            model_cfg = dict(type='CAINNet', norm='lys')
            MODELS.build(model_cfg).cuda()


def test_cain():

    # build model
    model = CAIN(
        generator=dict(type='CAINNet'),
        pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
        data_preprocessor=EditDataPreprocessor(
            input_view=(1, -1, 1, 1),
            output_view=(-1, 1, 1),
            pad_args=dict(mode='reflect')))

    # test attributes
    assert isinstance(model, CAIN)
    assert isinstance(model.data_preprocessor, EditDataPreprocessor)
    assert isinstance(model.generator, CAINNet)
    assert isinstance(model.pixel_loss, L1Loss)

    optimizer = Adam(model.generator.parameters(), lr=0.001)
    optim_wrapper = OptimWrapper(optimizer)

    # prepare data
    inputs = torch.rand(2, 3, 32, 32)
    target = torch.rand(3, 32, 32)
    data_sample = EditDataSample(gt_img=PixelData(data=target))
    data = [dict(inputs=inputs, data_sample=data_sample)]

    # train
    log_vars = model.train_step(data, optim_wrapper)
    assert isinstance(log_vars['loss'], torch.Tensor)

    # val
    output = model.val_step(data)
    assert output[0].pred_img.data.shape == (3, 32, 32)

    # feat
    output = model(torch.rand(1, 2, 3, 32, 32), mode='tensor')
    assert output.shape == (1, 3, 32, 32)
