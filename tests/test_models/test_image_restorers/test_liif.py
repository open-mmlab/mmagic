# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
from mmengine.optim import OptimWrapper
from torch.optim import Adam

from mmedit.data_element import EditDataSample, PixelData
from mmedit.models import LIIF, EditDataPreprocessor, build_backbone


def test_liif():

    model = LIIF(
        generator=dict(
            type='LIIFEDSRNet',
            encoder=dict(
                type='EDSRNet',
                in_channels=3,
                out_channels=3,
                mid_channels=4,
                num_blocks=2),
            imnet=dict(
                type='MLPRefiner',
                in_dim=64,
                out_dim=3,
                hidden_list=[4, 4, 4, 4]),
            local_ensemble=True,
            feat_unfold=True,
            cell_decode=True,
            eval_bsize=64),
        pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
        data_preprocessor=EditDataPreprocessor(
            mean=[0.4488 * 255, 0.4371 * 255, 0.4040 * 255],
            std=[255., 255., 255.],
            input_view=(-1, 1, 1),
            output_view=(1, -1)))

    # test attributes
    assert model.__class__.__name__ == 'LIIF'

    # prepare data
    inputs = torch.rand(3, 8, 8)
    data_sample = EditDataSample(
        metainfo=dict(coord=torch.rand(256, 2), cell=torch.rand(256, 2)))
    data_sample.gt_img = PixelData(data=torch.rand(256, 3))
    data = [dict(inputs=inputs, data_sample=data_sample)]

    optimizer = Adam(model.generator.parameters(), lr=0.001)
    optim_wrapper = OptimWrapper(optimizer)

    # train
    log_vars = model.train_step(data, optim_wrapper)
    assert isinstance(log_vars['loss'], torch.Tensor)
    save_loss = log_vars['loss']
    log_vars = model.train_step(data, optim_wrapper)
    log_vars = model.train_step(data, optim_wrapper)
    assert save_loss > log_vars['loss']

    # val
    predictions = model.val_step(data)
    assert isinstance(predictions, List)
    assert len(predictions) == 1
    assert isinstance(predictions[0], EditDataSample)
    assert isinstance(predictions[0].pred_img.data, torch.Tensor)
    assert predictions[0].pred_img.data.shape == (3, 16, 16)

    # feat
    output = model(torch.rand(1, 3, 8, 8), [data_sample], mode='tensor')
    assert output.shape == (1, 256, 3)


def test_liif_edsr_net():

    model_cfg = dict(
        type='LIIFEDSRNet',
        encoder=dict(
            type='EDSRNet',
            in_channels=3,
            out_channels=3,
            mid_channels=64,
            num_blocks=16),
        imnet=dict(
            type='MLPRefiner',
            in_dim=64,
            out_dim=3,
            hidden_list=[256, 256, 256, 256]),
        local_ensemble=True,
        feat_unfold=True,
        cell_decode=True,
        eval_bsize=30000)

    # build model
    model = build_backbone(model_cfg)

    # test attributes
    assert model.__class__.__name__ == 'LIIFEDSRNet'

    # prepare data
    inputs = torch.rand(1, 3, 22, 11)
    targets = torch.rand(1, 128 * 64, 3)
    coord = torch.rand(1, 128 * 64, 2)
    cell = torch.rand(1, 128 * 64, 2)

    # test on cpu
    output = model(inputs, coord, cell)
    output = model(inputs, coord, cell, True)
    assert torch.is_tensor(output)
    assert output.shape == targets.shape

    # test on gpu
    if torch.cuda.is_available():
        model = model.cuda()
        inputs = inputs.cuda()
        targets = targets.cuda()
        coord = coord.cuda()
        cell = cell.cuda()
        output = model(inputs, coord, cell)
        output = model(inputs, coord, cell, True)
        assert torch.is_tensor(output)
        assert output.shape == targets.shape


def test_liif_rdn_net():

    model_cfg = dict(
        type='LIIFRDNNet',
        encoder=dict(
            type='RDNNet',
            in_channels=3,
            out_channels=3,
            mid_channels=64,
            num_blocks=16,
            upscale_factor=4,
            num_layers=8,
            channel_growth=64),
        imnet=dict(
            type='MLPRefiner',
            in_dim=64,
            out_dim=3,
            hidden_list=[256, 256, 256, 256]),
        local_ensemble=True,
        feat_unfold=True,
        cell_decode=True,
        eval_bsize=30000)

    # build model
    model = build_backbone(model_cfg)

    # test attributes
    assert model.__class__.__name__ == 'LIIFRDNNet'

    # prepare data
    inputs = torch.rand(1, 3, 22, 11)
    targets = torch.rand(1, 128 * 64, 3)
    coord = torch.rand(1, 128 * 64, 2)
    cell = torch.rand(1, 128 * 64, 2)

    # test on cpu
    output = model(inputs, coord, cell)
    output = model(inputs, coord, cell, True)
    assert torch.is_tensor(output)
    assert output.shape == targets.shape

    # test on gpu
    if torch.cuda.is_available():
        model = model.cuda()
        inputs = inputs.cuda()
        targets = targets.cuda()
        coord = coord.cuda()
        cell = cell.cuda()
        output = model(inputs, coord, cell)
        output = model(inputs, coord, cell, True)
        assert torch.is_tensor(output)
        assert output.shape == targets.shape
