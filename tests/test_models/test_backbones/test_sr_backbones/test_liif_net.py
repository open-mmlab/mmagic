# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmedit.models import build_backbone


def test_liif_edsr():

    model_cfg = dict(
        type='LIIFEDSR',
        encoder=dict(
            type='EDSR',
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
    assert model.__class__.__name__ == 'LIIFEDSR'

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


def test_liif_rdn():

    model_cfg = dict(
        type='LIIFRDN',
        encoder=dict(
            type='RDN',
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
    assert model.__class__.__name__ == 'LIIFRDN'

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
