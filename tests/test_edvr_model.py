import mmcv
import pytest
import torch
from mmcv.runner import obj_from_dict
from mmedit.models import build_model
from mmedit.models.backbones import EDVRNet
from mmedit.models.losses import L1Loss


def test_edvr_model():

    model_cfg = dict(
        type='EDVR',
        generator=dict(
            type='EDVRNet',
            in_channels=3,
            out_channels=3,
            mid_channels=8,
            num_frames=5,
            deformable_groups=2,
            num_blocks_extraction=1,
            num_blocks_reconstruction=1,
            center_frame_idx=2,
            with_tsa=False),
        pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='sum'),
    )

    train_cfg = None
    test_cfg = None

    # build restorer
    restorer = build_model(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)

    # test attributes
    assert restorer.__class__.__name__ == 'EDVR'
    assert isinstance(restorer.generator, EDVRNet)
    assert isinstance(restorer.pixel_loss, L1Loss)

    # prepare data
    inputs = torch.rand(1, 5, 3, 8, 8)
    targets = torch.rand(1, 3, 32, 32)

    # test train_step and forward_test (gpu)
    if torch.cuda.is_available():
        restorer = restorer.cuda()
        data_batch = {'lq': inputs.cuda(), 'gt': targets.cuda()}

        # prepare optimizer
        optim_cfg = dict(type='Adam', lr=2e-4, betas=(0.9, 0.999))
        optimizer = {
            'generator':
            obj_from_dict(
                optim_cfg, torch.optim,
                dict(params=getattr(restorer, 'generator').parameters()))
        }

        # train_step
        outputs = restorer.train_step(data_batch, optimizer)
        assert isinstance(outputs, dict)
        assert isinstance(outputs['log_vars'], dict)
        assert isinstance(outputs['log_vars']['loss_pix'], float)
        assert outputs['num_samples'] == 1
        assert torch.equal(outputs['results']['lq'], data_batch['lq'].cpu())
        assert torch.equal(outputs['results']['gt'], data_batch['gt'].cpu())
        assert torch.is_tensor(outputs['results']['output'])
        assert outputs['results']['output'].size() == (1, 3, 32, 32)

        # with TSA
        model_cfg['generator']['with_tsa'] = True
        train_cfg = mmcv.ConfigDict(tsa_iter=1)
        restorer = build_model(
            model_cfg, train_cfg=train_cfg, test_cfg=test_cfg).cuda()
        optimizer = {
            'generator':
            obj_from_dict(
                optim_cfg, torch.optim,
                dict(params=getattr(restorer, 'generator').parameters()))
        }
        # train without updating tsa module
        outputs = restorer.train_step(data_batch, optimizer)
        # train with updating tsa module
        outputs = restorer.train_step(data_batch, optimizer)
        assert isinstance(outputs, dict)
        assert isinstance(outputs['log_vars'], dict)
        assert isinstance(outputs['log_vars']['loss_pix'], float)
        assert outputs['num_samples'] == 1
        assert torch.equal(outputs['results']['lq'], data_batch['lq'].cpu())
        assert torch.equal(outputs['results']['gt'], data_batch['gt'].cpu())
        assert torch.is_tensor(outputs['results']['output'])
        assert outputs['results']['output'].size() == (1, 3, 32, 32)

        with pytest.raises(KeyError):
            # In TSA mode, train_cfg must contain "tsa_iter"
            train_cfg = None
            restorer = build_model(
                model_cfg, train_cfg=train_cfg, test_cfg=test_cfg).cuda()

            train_cfg = mmcv.ConfigDict(other_content='xxx')
            restorer = build_model(
                model_cfg, train_cfg=train_cfg, test_cfg=test_cfg).cuda()
