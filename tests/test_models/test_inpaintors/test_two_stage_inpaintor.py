# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import tempfile

import pytest
import torch
from mmcv import Config

from mmedit.core import build_optimizers
from mmedit.models import TwoStageInpaintor


def test_two_stage_inpaintor():
    model = dict(
        disc_input_with_mask=True,
        encdec=dict(
            type='DeepFillEncoderDecoder',
            stage1=dict(
                type='GLEncoderDecoder',
                encoder=dict(
                    type='DeepFillEncoder',
                    conv_type='gated_conv',
                    channel_factor=0.75),
                decoder=dict(
                    type='DeepFillDecoder',
                    conv_type='gated_conv',
                    in_channels=96,
                    channel_factor=0.75),
                dilation_neck=dict(
                    type='GLDilationNeck',
                    in_channels=96,
                    conv_type='gated_conv',
                    act_cfg=dict(type='ELU'))),
            stage2=dict(
                type='DeepFillRefiner',
                encoder_attention=dict(
                    type='DeepFillEncoder',
                    encoder_type='stage2_attention',
                    conv_type='gated_conv',
                    channel_factor=0.75),
                encoder_conv=dict(
                    type='DeepFillEncoder',
                    encoder_type='stage2_conv',
                    conv_type='gated_conv',
                    channel_factor=0.75),
                dilation_neck=dict(
                    type='GLDilationNeck',
                    in_channels=96,
                    conv_type='gated_conv',
                    act_cfg=dict(type='ELU')),
                contextual_attention=dict(
                    type='ContextualAttentionNeck',
                    in_channels=96,
                    conv_type='gated_conv'),
                decoder=dict(
                    type='DeepFillDecoder',
                    in_channels=192,
                    conv_type='gated_conv'))),
        disc=dict(
            type='MultiLayerDiscriminator',
            in_channels=4,
            max_channels=256,
            fc_in_channels=256 * 4 * 4,
            fc_out_channels=1,
            num_convs=6,
            norm_cfg=None,
            act_cfg=dict(type='ELU'),
            out_act_cfg=dict(type='LeakyReLU', negative_slope=0.2),
            with_spectral_norm=True,
        ),
        stage1_loss_type=('loss_l1_hole', 'loss_l1_valid'),
        stage2_loss_type=('loss_l1_hole', 'loss_l1_valid', 'loss_gan'),
        loss_gan=dict(
            type='GANLoss',
            gan_type='hinge',
            loss_weight=1,
        ),
        loss_l1_hole=dict(
            type='L1Loss',
            loss_weight=1.0,
        ),
        loss_l1_valid=dict(
            type='L1Loss',
            loss_weight=1.0,
        ),
        pretrained=None)

    train_cfg = Config(dict(disc_step=1))
    test_cfg = Config(dict(metrics=['l1', 'psnr', 'ssim']))

    tsinpaintor = TwoStageInpaintor(
        **model, train_cfg=train_cfg, test_cfg=test_cfg)

    # check architecture
    assert tsinpaintor.stage1_loss_type == ('loss_l1_hole', 'loss_l1_valid')
    assert tsinpaintor.stage2_loss_type == ('loss_l1_hole', 'loss_l1_valid',
                                            'loss_gan')
    assert tsinpaintor.with_l1_hole_loss
    assert tsinpaintor.with_l1_valid_loss
    assert not tsinpaintor.with_composed_percep_loss
    assert not tsinpaintor.with_out_percep_loss
    assert tsinpaintor.with_gan

    if torch.cuda.is_available():
        # prepare data
        gt_img = torch.rand((2, 3, 256, 256)).cuda()
        mask = torch.zeros((2, 1, 256, 256)).cuda()
        mask[..., 50:180, 60:170] = 1.
        masked_img = gt_img * (1. - mask)

        data_batch = dict(gt_img=gt_img, mask=mask, masked_img=masked_img)

        # prepare model and optimizer
        tsinpaintor.cuda()
        optimizers_config = dict(
            generator=dict(type='Adam', lr=0.0001),
            disc=dict(type='Adam', lr=0.0001))

        optims = build_optimizers(tsinpaintor, optimizers_config)

        # check train_step with standard deepfillv2 model
        outputs = tsinpaintor.train_step(data_batch, optims)

        assert outputs['num_samples'] == 2
        log_vars = outputs['log_vars']
        assert 'real_loss' in log_vars
        assert 'stage1_loss_l1_hole' in log_vars
        assert 'stage1_loss_l1_valid' in log_vars
        assert 'stage2_loss_l1_hole' in log_vars
        assert 'stage2_loss_l1_valid' in log_vars
        assert 'stage1_fake_res' in outputs['results']
        assert 'stage2_fake_res' in outputs['results']
        assert outputs['results']['stage1_fake_res'].size() == (2, 3, 256, 256)

        # check train step w/o disc step
        tsinpaintor.train_cfg.disc_step = 0
        outputs = tsinpaintor.train_step(data_batch, optims)

        assert outputs['num_samples'] == 2
        log_vars = outputs['log_vars']
        assert 'real_loss' not in log_vars
        assert 'stage1_loss_l1_hole' in log_vars
        assert 'stage1_loss_l1_valid' in log_vars
        assert 'stage2_loss_l1_hole' in log_vars
        assert 'stage2_loss_l1_valid' in log_vars
        assert 'stage1_fake_res' in outputs['results']
        assert 'stage2_fake_res' in outputs['results']
        assert outputs['results']['stage1_fake_res'].size() == (2, 3, 256, 256)
        tsinpaintor.train_cfg.disc_step = 1

        # check train step w/ multiple disc step
        tsinpaintor.train_cfg.disc_step = 5
        outputs = tsinpaintor.train_step(data_batch, optims)

        assert outputs['num_samples'] == 2
        log_vars = outputs['log_vars']
        assert 'real_loss' in log_vars
        assert 'stage1_loss_l1_hole' not in log_vars
        assert outputs['results']['fake_res'].size() == (2, 3, 256, 256)
        tsinpaintor.train_cfg.disc_step = 1

        # test forward test w/o save image
        outputs = tsinpaintor.forward_test(
            masked_img[0:1], mask[0:1], gt_img=gt_img[0:1, ...])
        assert 'eval_result' in outputs
        assert outputs['eval_result']['l1'] > 0
        assert outputs['eval_result']['psnr'] > 0
        assert outputs['eval_result']['ssim'] > 0

        # test forward test w/o eval metrics
        tsinpaintor.test_cfg = dict()
        tsinpaintor.eval_with_metrics = False
        outputs = tsinpaintor.forward_test(masked_img[0:1], mask[0:1])
        for key in [
                'stage1_fake_res', 'stage2_fake_res', 'fake_res', 'fake_img'
        ]:
            assert outputs[key].size() == (1, 3, 256, 256)

        # test forward test w/ save image
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = tsinpaintor.forward_test(
                masked_img[0:1],
                mask[0:1],
                save_image=True,
                save_path=tmpdir,
                iteration=4396,
                meta=[dict(gt_img_path='igccc.png')])

            assert os.path.exists(os.path.join(tmpdir, 'igccc_4396.png'))

        # test forward test w/ save image w/ gt_img
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = tsinpaintor.forward_test(
                masked_img[0:1],
                mask[0:1],
                save_image=True,
                save_path=tmpdir,
                meta=[dict(gt_img_path='igccc.png')],
                gt_img=gt_img[0:1, ...])

            assert os.path.exists(os.path.join(tmpdir, 'igccc.png'))

            with pytest.raises(AssertionError):
                outputs = tsinpaintor.forward_test(
                    masked_img[0:1],
                    mask[0:1],
                    save_image=True,
                    save_path=tmpdir,
                    iteration=4396,
                    gt_img=gt_img[0:1, ...])
            with pytest.raises(AssertionError):
                outputs = tsinpaintor.forward_test(
                    masked_img[0:1],
                    mask[0:1],
                    save_image=True,
                    save_path=None,
                    iteration=4396,
                    meta=[dict(gt_img_path='igccc.png')],
                    gt_img=gt_img[0:1, ...])

        # check train_step with not implemented loss type
        with pytest.raises(NotImplementedError):
            model_ = copy.deepcopy(model)
            model_['stage1_loss_type'] = ('igccc', )
            tsinpaintor = TwoStageInpaintor(
                **model_, train_cfg=train_cfg, test_cfg=test_cfg).cuda()
            outputs = tsinpaintor.train_step(data_batch, optims)

        # test input w/o ones and disc input w/o mask
        model_ = dict(
            disc_input_with_mask=False,
            input_with_ones=False,
            encdec=dict(
                type='DeepFillEncoderDecoder',
                stage1=dict(
                    type='GLEncoderDecoder',
                    encoder=dict(
                        type='DeepFillEncoder',
                        in_channels=4,
                        conv_type='gated_conv',
                        channel_factor=0.75),
                    decoder=dict(
                        type='DeepFillDecoder',
                        conv_type='gated_conv',
                        in_channels=96,
                        channel_factor=0.75),
                    dilation_neck=dict(
                        type='GLDilationNeck',
                        in_channels=96,
                        conv_type='gated_conv',
                        act_cfg=dict(type='ELU'))),
                stage2=dict(
                    type='DeepFillRefiner',
                    encoder_attention=dict(
                        type='DeepFillEncoder',
                        in_channels=4,
                        encoder_type='stage2_attention',
                        conv_type='gated_conv',
                        channel_factor=0.75),
                    encoder_conv=dict(
                        type='DeepFillEncoder',
                        in_channels=4,
                        encoder_type='stage2_conv',
                        conv_type='gated_conv',
                        channel_factor=0.75),
                    dilation_neck=dict(
                        type='GLDilationNeck',
                        in_channels=96,
                        conv_type='gated_conv',
                        act_cfg=dict(type='ELU')),
                    contextual_attention=dict(
                        type='ContextualAttentionNeck',
                        in_channels=96,
                        conv_type='gated_conv'),
                    decoder=dict(
                        type='DeepFillDecoder',
                        in_channels=192,
                        conv_type='gated_conv'))),
            disc=dict(
                type='MultiLayerDiscriminator',
                in_channels=3,
                max_channels=256,
                fc_in_channels=256 * 4 * 4,
                fc_out_channels=1,
                num_convs=6,
                norm_cfg=None,
                act_cfg=dict(type='ELU'),
                out_act_cfg=dict(type='LeakyReLU', negative_slope=0.2),
                with_spectral_norm=True,
            ),
            stage1_loss_type=('loss_l1_hole', 'loss_l1_valid'),
            stage2_loss_type=('loss_l1_hole', 'loss_l1_valid', 'loss_gan'),
            loss_gan=dict(
                type='GANLoss',
                gan_type='hinge',
                loss_weight=1,
            ),
            loss_l1_hole=dict(
                type='L1Loss',
                loss_weight=1.0,
            ),
            loss_gp=dict(type='GradientPenaltyLoss', loss_weight=10.),
            loss_tv=dict(
                type='MaskedTVLoss',
                loss_weight=0.1,
            ),
            loss_l1_valid=dict(
                type='L1Loss',
                loss_weight=1.0,
            ),
            pretrained=None)
        tsinpaintor = TwoStageInpaintor(
            **model_, train_cfg=train_cfg, test_cfg=test_cfg).cuda()
        outputs = tsinpaintor.train_step(data_batch, optims)

        assert outputs['num_samples'] == 2
        log_vars = outputs['log_vars']
        assert 'real_loss' in log_vars
        assert 'stage1_loss_l1_hole' in log_vars
        assert 'stage1_loss_l1_valid' in log_vars
        assert 'stage2_loss_l1_hole' in log_vars
        assert 'stage2_loss_l1_valid' in log_vars
        assert 'stage1_fake_res' in outputs['results']
        assert 'stage2_fake_res' in outputs['results']
        assert outputs['results']['stage1_fake_res'].size() == (2, 3, 256, 256)

        outputs = tsinpaintor.forward_test(
            masked_img[0:1], mask[0:1], gt_img=gt_img[0:1, ...])
        assert 'eval_result' in outputs
        assert outputs['eval_result']['l1'] > 0

        # test w/o stage1 loss
        model_ = copy.deepcopy(model)
        model_['stage1_loss_type'] = None
        tsinpaintor = TwoStageInpaintor(
            **model_, train_cfg=train_cfg, test_cfg=test_cfg).cuda()
        outputs = tsinpaintor.train_step(data_batch, optims)

        assert outputs['num_samples'] == 2
        log_vars = outputs['log_vars']
        assert 'real_loss' in log_vars
        assert 'stage1_loss_l1_hole' not in log_vars
        assert 'stage1_loss_l1_valid' not in log_vars
        assert 'stage2_loss_l1_hole' in log_vars
        assert 'stage2_loss_l1_valid' in log_vars
        assert 'stage1_fake_res' in outputs['results']
        assert 'stage2_fake_res' in outputs['results']
        assert outputs['results']['stage1_fake_res'].size() == (2, 3, 256, 256)

        # test w/o stage2 loss
        model_ = copy.deepcopy(model)
        model_['stage2_loss_type'] = None
        tsinpaintor = TwoStageInpaintor(
            **model_, train_cfg=train_cfg, test_cfg=test_cfg).cuda()
        outputs = tsinpaintor.train_step(data_batch, optims)

        assert outputs['num_samples'] == 2
        log_vars = outputs['log_vars']
        assert 'real_loss' in log_vars
        assert 'stage1_loss_l1_hole' in log_vars
        assert 'stage1_loss_l1_valid' in log_vars
        assert 'stage2_loss_l1_hole' not in log_vars
        assert 'stage2_loss_l1_valid' not in log_vars
        assert 'stage1_fake_res' in outputs['results']
        assert 'stage2_fake_res' in outputs['results']
        assert outputs['results']['stage1_fake_res'].size() == (2, 3, 256, 256)
