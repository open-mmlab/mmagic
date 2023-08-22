# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from unittest.mock import patch

import mmcv
import numpy as np
import pytest
import torch

from mmedit.models import BaseMattor, build_model


def _get_model_cfg(fname):
    """Grab configs necessary to create a model.

    These are deep copied to allow for safe modification of parameters without
    influencing other tests.
    """
    config_dpath = 'configs/mattors'
    config_fpath = osp.join(config_dpath, fname)
    if not osp.exists(config_dpath):
        raise Exception('Cannot find config path')
    config = mmcv.Config.fromfile(config_fpath)
    return config.model, config.train_cfg, config.test_cfg


def assert_dict_keys_equal(dictionary, target_keys):
    """Check if the keys of the dictionary is equal to the target key set."""
    assert isinstance(dictionary, dict)
    assert set(dictionary.keys()) == set(target_keys)


@patch.multiple(BaseMattor, __abstractmethods__=set())
def test_base_mattor():
    backbone = dict(
        type='SimpleEncoderDecoder',
        encoder=dict(type='VGG16', in_channels=4),
        decoder=dict(type='PlainDecoder'))
    refiner = dict(type='PlainRefiner')
    train_cfg = mmcv.ConfigDict(train_backbone=True, train_refiner=True)
    test_cfg = mmcv.ConfigDict(
        refine=True, metrics=['SAD', 'MSE', 'GRAD', 'CONN'])

    with pytest.raises(KeyError):
        # metrics should be specified in test_cfg
        BaseMattor(
            backbone,
            refiner,
            train_cfg.copy(),
            test_cfg=mmcv.ConfigDict(refine=True))

    with pytest.raises(KeyError):
        # supported metric should be one of {'SAD', 'MSE'}
        BaseMattor(
            backbone,
            refiner,
            train_cfg.copy(),
            test_cfg=mmcv.ConfigDict(
                refine=True, metrics=['UnsupportedMetric']))

    with pytest.raises(TypeError):
        # metrics must be None or a list of str
        BaseMattor(
            backbone,
            refiner,
            train_cfg.copy(),
            test_cfg=mmcv.ConfigDict(refine=True, metrics='SAD'))

    # build mattor without refiner
    mattor = BaseMattor(
        backbone, refiner=None, train_cfg=None, test_cfg=test_cfg.copy())
    assert not mattor.with_refiner

    # only train the refiner, this will freeze the backbone
    mattor = BaseMattor(
        backbone,
        refiner,
        train_cfg=mmcv.ConfigDict(train_backbone=False, train_refiner=True),
        test_cfg=test_cfg.copy())
    assert not mattor.train_cfg.train_backbone
    assert mattor.train_cfg.train_refiner
    assert mattor.test_cfg.refine

    # only train the backbone while the refiner is used for inference but not
    # trained, this behavior is allowed currently but will cause a warning.
    mattor = BaseMattor(
        backbone,
        refiner,
        train_cfg=mmcv.ConfigDict(train_backbone=True, train_refiner=False),
        test_cfg=test_cfg.copy())
    assert mattor.train_cfg.train_backbone
    assert not mattor.train_cfg.train_refiner
    assert mattor.test_cfg.refine


def test_dim():
    model_cfg, train_cfg, test_cfg = _get_model_cfg(
        'dim/dim_stage3_v16_pln_1x1_1000k_comp1k.py')
    model_cfg['pretrained'] = None

    # 1. test dim model with refiner
    train_cfg.train_refiner = True
    test_cfg.refine = True

    # test model forward in train mode
    model = build_model(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)
    input_train = _demo_input_train((64, 64))
    output_train = model(**input_train)
    assert output_train['num_samples'] == 1
    assert_dict_keys_equal(output_train['losses'],
                           ['loss_alpha', 'loss_comp', 'loss_refine'])

    # test model forward in train mode with gpu
    if torch.cuda.is_available():
        model = build_model(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)
        model.cuda()
        input_train = _demo_input_train((64, 64), cuda=True)
        output_train = model(**input_train)
        assert output_train['num_samples'] == 1
        assert_dict_keys_equal(output_train['losses'],
                               ['loss_alpha', 'loss_comp', 'loss_refine'])

    # test model forward in test mode
    with torch.no_grad():
        model = build_model(model_cfg, train_cfg=None, test_cfg=test_cfg)
        input_test = _demo_input_test((64, 64))
        output_test = model(**input_test, test_mode=True)
        assert isinstance(output_test['pred_alpha'], np.ndarray)
        assert_dict_keys_equal(output_test['eval_result'],
                               ['SAD', 'MSE', 'GRAD', 'CONN'])

        # test model forward in test mode with gpu
        if torch.cuda.is_available():
            model = build_model(model_cfg, train_cfg=None, test_cfg=test_cfg)
            model.cuda()
            input_test = _demo_input_test((64, 64), cuda=True)
            output_test = model(**input_test, test_mode=True)
            assert isinstance(output_test['pred_alpha'], np.ndarray)
            assert_dict_keys_equal(output_test['eval_result'],
                                   ['SAD', 'MSE', 'GRAD', 'CONN'])

    # 2. test dim model without refiner
    model_cfg['refiner'] = None
    test_cfg['metrics'] = None

    # test model forward in train mode
    model = build_model(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)
    input_train = _demo_input_train((64, 64))
    output_train = model(**input_train)
    assert output_train['num_samples'] == 1
    assert_dict_keys_equal(output_train['losses'], ['loss_alpha', 'loss_comp'])

    # test model forward in train mode with gpu
    if torch.cuda.is_available():
        model = build_model(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)
        model.cuda()
        input_train = _demo_input_train((64, 64), cuda=True)
        output_train = model(**input_train)
        assert output_train['num_samples'] == 1
        assert_dict_keys_equal(output_train['losses'],
                               ['loss_alpha', 'loss_comp'])

    # test model forward in test mode
    with torch.no_grad():
        model = build_model(model_cfg, train_cfg=None, test_cfg=test_cfg)
        input_test = _demo_input_test((64, 64))
        output_test = model(**input_test, test_mode=True)
        assert isinstance(output_test['pred_alpha'], np.ndarray)
        assert output_test['eval_result'] is None

        # check test with gpu
        if torch.cuda.is_available():
            model = build_model(model_cfg, train_cfg=None, test_cfg=test_cfg)
            model.cuda()
            input_test = _demo_input_test((64, 64), cuda=True)
            output_test = model(**input_test, test_mode=True)
            assert isinstance(output_test['pred_alpha'], np.ndarray)
            assert output_test['eval_result'] is None

    # test forward_dummy
    model.cpu().eval()
    inputs = torch.ones((1, 4, 32, 32))
    model.forward_dummy(inputs)


def test_indexnet():
    model_cfg, _, test_cfg = _get_model_cfg(
        'indexnet/indexnet_mobv2_1x16_78k_comp1k.py')
    model_cfg['pretrained'] = None

    # test indexnet inference
    with torch.no_grad():
        indexnet = build_model(model_cfg, train_cfg=None, test_cfg=test_cfg)
        indexnet.eval()
        input_test = _demo_input_test((64, 64))
        output_test = indexnet(**input_test, test_mode=True)
        assert isinstance(output_test['pred_alpha'], np.ndarray)
        assert output_test['pred_alpha'].shape == (64, 64)
        assert_dict_keys_equal(output_test['eval_result'],
                               ['SAD', 'MSE', 'GRAD', 'CONN'])

        # test inference with gpu
        if torch.cuda.is_available():
            indexnet = build_model(
                model_cfg, train_cfg=None, test_cfg=test_cfg).cuda()
            indexnet.eval()
            input_test = _demo_input_test((64, 64), cuda=True)
            output_test = indexnet(**input_test, test_mode=True)
            assert isinstance(output_test['pred_alpha'], np.ndarray)
            assert output_test['pred_alpha'].shape == (64, 64)
            assert_dict_keys_equal(output_test['eval_result'],
                                   ['SAD', 'MSE', 'GRAD', 'CONN'])

    # test forward train though we do not guarantee the training for present
    model_cfg.loss_alpha = None
    model_cfg.loss_comp = dict(type='L1CompositionLoss')
    indexnet = build_model(
        model_cfg,
        train_cfg=mmcv.ConfigDict(train_backbone=True),
        test_cfg=test_cfg)
    input_train = _demo_input_train((64, 64), batch_size=2)
    output_train = indexnet(**input_train)
    assert output_train['num_samples'] == 2
    assert_dict_keys_equal(output_train['losses'], ['loss_comp'])

    if torch.cuda.is_available():
        model_cfg.loss_alpha = dict(type='L1Loss')
        model_cfg.loss_comp = None
        indexnet = build_model(
            model_cfg,
            train_cfg=mmcv.ConfigDict(train_backbone=True),
            test_cfg=test_cfg).cuda()
        input_train = _demo_input_train((64, 64), batch_size=2, cuda=True)
        output_train = indexnet(**input_train)
        assert output_train['num_samples'] == 2
        assert_dict_keys_equal(output_train['losses'], ['loss_alpha'])

    # test forward_dummy
    indexnet.cpu().eval()
    inputs = torch.ones((1, 4, 32, 32))
    indexnet.forward_dummy(inputs)


def test_gca():
    model_cfg, train_cfg, test_cfg = _get_model_cfg(
        'gca/gca_r34_4x10_200k_comp1k.py')
    model_cfg['pretrained'] = None

    # test model forward in train mode
    model = build_model(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)
    inputs = _demo_input_train((64, 64), batch_size=2)
    inputs['trimap'] = inputs['trimap'].expand_as(inputs['merged'])
    inputs['meta'][0]['to_onehot'] = True
    outputs = model(inputs['merged'], inputs['trimap'], inputs['meta'],
                    inputs['alpha'])
    assert outputs['num_samples'] == 2
    assert_dict_keys_equal(outputs['losses'], ['loss'])

    if torch.cuda.is_available():
        model = build_model(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)
        model.cuda()
        inputs = _demo_input_train((64, 64), batch_size=2, cuda=True)
        inputs['trimap'] = inputs['trimap'].expand_as(inputs['merged'])
        inputs['meta'][0]['to_onehot'] = True
        outputs = model(inputs['merged'], inputs['trimap'], inputs['meta'],
                        inputs['alpha'])
        assert outputs['num_samples'] == 2
        assert_dict_keys_equal(outputs['losses'], ['loss'])

    # test model forward in test mode
    with torch.no_grad():
        model_cfg.backbone.encoder.in_channels = 4
        model = build_model(model_cfg, train_cfg=None, test_cfg=test_cfg)
        inputs = _demo_input_test((64, 64))
        outputs = model(**inputs, test_mode=True)
        assert_dict_keys_equal(outputs['eval_result'],
                               ['SAD', 'MSE', 'GRAD', 'CONN'])

        if torch.cuda.is_available():
            model = build_model(model_cfg, train_cfg=None, test_cfg=test_cfg)
            model.cuda()
            inputs = _demo_input_test((64, 64), cuda=True)
            outputs = model(**inputs, test_mode=True)
            assert_dict_keys_equal(outputs['eval_result'],
                                   ['SAD', 'MSE', 'GRAD', 'CONN'])

    # test forward_dummy
    model.cpu().eval()
    inputs = torch.ones((1, 4, 32, 32))
    model.forward_dummy(inputs)


def _demo_input_train(img_shape, batch_size=1, cuda=False):
    """Create a superset of inputs needed to run backbone.

    Args:
        img_shape (tuple): shape of the input image.
        batch_size (int): batch size of the input batch.
        cuda (bool): whether transfer input into gpu.
    """
    color_shape = (batch_size, 3, img_shape[0], img_shape[1])
    gray_shape = (batch_size, 1, img_shape[0], img_shape[1])
    merged = torch.from_numpy(np.random.random(color_shape).astype(np.float32))
    trimap = torch.from_numpy(
        np.random.randint(255, size=gray_shape).astype(np.float32))
    meta = [{}] * batch_size
    alpha = torch.from_numpy(np.random.random(gray_shape).astype(np.float32))
    ori_merged = torch.from_numpy(
        np.random.random(color_shape).astype(np.float32))
    fg = torch.from_numpy(np.random.random(color_shape).astype(np.float32))
    bg = torch.from_numpy(np.random.random(color_shape).astype(np.float32))
    if cuda:
        merged = merged.cuda()
        trimap = trimap.cuda()
        alpha = alpha.cuda()
        ori_merged = ori_merged.cuda()
        fg = fg.cuda()
        bg = bg.cuda()

    return dict(
        merged=merged,
        trimap=trimap,
        meta=meta,
        alpha=alpha,
        ori_merged=ori_merged,
        fg=fg,
        bg=bg)


def _demo_input_test(img_shape, batch_size=1, cuda=False, test_trans='resize'):
    """Create a superset of inputs needed to run backbone.

    Args:
        img_shape (tuple): shape of the input image.
        batch_size (int): batch size of the input batch.
        cuda (bool): whether transfer input into gpu.
        test_trans (str): what test transformation is used in data pipeline.
    """
    color_shape = (batch_size, 3, img_shape[0], img_shape[1])
    gray_shape = (batch_size, 1, img_shape[0], img_shape[1])
    merged = torch.from_numpy(np.random.random(color_shape).astype(np.float32))
    trimap = torch.from_numpy(
        np.random.randint(255, size=gray_shape).astype(np.float32))
    ori_alpha = np.random.random(img_shape).astype(np.float32)
    ori_trimap = np.random.randint(256, size=img_shape).astype(np.float32)
    if cuda:
        merged = merged.cuda()
        trimap = trimap.cuda()
    meta = [
        dict(
            ori_alpha=ori_alpha,
            ori_trimap=ori_trimap,
            merged_ori_shape=img_shape)
    ] * batch_size

    if test_trans == 'pad':
        meta[0]['pad'] = (0, 0)
    elif test_trans == 'resize':
        # we just test bilinear as the interpolation method
        meta[0]['interpolation'] = 'bilinear'

    return dict(merged=merged, trimap=trimap, meta=meta)
