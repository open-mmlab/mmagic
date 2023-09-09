# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmagic.registry import MODELS


@pytest.mark.skipif(
    'win' in platform.system().lower() and 'cu' in torch.__version__,
    reason='skip on windows-cuda due to limited RAM.')
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
    model = MODELS.build(model_cfg)

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
    model = MODELS.build(model_cfg)

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


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
