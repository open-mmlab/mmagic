# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
from mmcv.runner import obj_from_dict
from mmcv.utils.config import Config

from mmedit.models import build_model
from mmedit.models.registry import COMPONENTS


@COMPONENTS.register_module()
class BP(nn.Module):
    """A simple BP network for testing LIIF.

    Args:
        in_dim (int): Input dimension.
        out_dim (int): Output dimension.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layer = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layer(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)


def test_liif():

    model_cfg = dict(
        type='LIIF',
        generator=dict(
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
            eval_bsize=30000),
        rgb_mean=(0.4488, 0.4371, 0.4040),
        rgb_std=(1., 1., 1.),
        pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'))

    scale_max = 4
    train_cfg = None
    test_cfg = Config(dict(metrics=['PSNR', 'SSIM'], crop_border=scale_max))

    # build restorer
    restorer = build_model(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)

    # test attributes
    assert restorer.__class__.__name__ == 'LIIF'

    # prepare data
    inputs = torch.rand(1, 3, 22, 11)
    targets = torch.rand(1, 128 * 64, 3)
    coord = torch.rand(1, 128 * 64, 2)
    cell = torch.rand(1, 128 * 64, 2)
    data_batch = {'lq': inputs, 'gt': targets, 'coord': coord, 'cell': cell}

    # prepare optimizer
    optim_cfg = dict(type='Adam', lr=1e-4, betas=(0.9, 0.999))
    optimizer = obj_from_dict(optim_cfg, torch.optim,
                              dict(params=restorer.parameters()))

    # test train_step and forward_test (cpu)
    outputs = restorer.train_step(data_batch, optimizer)
    assert isinstance(outputs, dict)
    assert isinstance(outputs['log_vars'], dict)
    assert isinstance(outputs['log_vars']['loss_pix'], float)
    assert outputs['num_samples'] == 1
    assert outputs['results']['lq'].shape == data_batch['lq'].shape
    assert outputs['results']['gt'].shape == data_batch['gt'].shape
    assert torch.is_tensor(outputs['results']['output'])
    assert outputs['results']['output'].size() == (1, 128 * 64, 3)

    # test train_step and forward_test (gpu)
    if torch.cuda.is_available():
        restorer = restorer.cuda()
        data_batch = {
            'lq': inputs.cuda(),
            'gt': targets.cuda(),
            'coord': coord.cuda(),
            'cell': cell.cuda()
        }

        # train_step
        optimizer = obj_from_dict(optim_cfg, torch.optim,
                                  dict(params=restorer.parameters()))
        outputs = restorer.train_step(data_batch, optimizer)
        assert isinstance(outputs, dict)
        assert isinstance(outputs['log_vars'], dict)
        assert isinstance(outputs['log_vars']['loss_pix'], float)
        assert outputs['num_samples'] == 1
        assert outputs['results']['lq'].shape == data_batch['lq'].shape
        assert outputs['results']['gt'].shape == data_batch['gt'].shape
        assert torch.is_tensor(outputs['results']['output'])
        assert outputs['results']['output'].size() == (1, 128 * 64, 3)

        # val_step
        result = restorer.val_step(data_batch, meta=[{'gt_path': ''}])
        assert isinstance(result, dict)
        assert isinstance(result['eval_result'], dict)
        assert result['eval_result'].keys() == set({'PSNR', 'SSIM'})
        assert isinstance(result['eval_result']['PSNR'], np.float64)
        assert isinstance(result['eval_result']['SSIM'], np.float64)
