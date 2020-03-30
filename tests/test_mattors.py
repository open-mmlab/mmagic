import os.path as osp

import mmcv
import numpy as np
import torch
from mmedit.models import build_model


def _get_model_cfg(fname):
    """
    Grab configs necessary to create a model. These are deep copied to allow
    for safe modification of parameters without influencing other tests.
    """
    repo_dpath = osp.dirname(osp.dirname(__file__))
    config_dpath = osp.join(repo_dpath, 'configs/mattors')
    config_fpath = osp.join(config_dpath, fname)
    if not osp.exists(config_dpath):
        raise Exception('Cannot find config path')
    config = mmcv.Config.fromfile(config_fpath)
    return config.model, config.train_cfg, config.test_cfg


def assert_dict_keys_equal(dictionary, target_keys):
    """Check if the keys of the dictionary is equal to the target key set."""
    assert isinstance(dictionary, dict)
    assert set(dictionary.keys()) == set(target_keys)


def test_dim():
    model_cfg, train_cfg, test_cfg = _get_model_cfg('dim_stage3.py')
    model_cfg['pretrained'] = None

    # 1. test dim model with refiner
    train_cfg.train_refiner = True
    test_cfg.refine = True

    # test model forward in train mode
    model = build_model(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)
    input_train = _demo_input_train((64, 64))
    losses = model(**input_train)
    assert_dict_keys_equal(losses, ['loss_alpha', 'loss_comp', 'loss_refine'])

    # test model forward in train mode with gpu
    if torch.cuda.is_available():
        model = build_model(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)
        model.cuda()
        input_train = _demo_input_train((64, 64), cuda=True)
        losses = model(**input_train)
        assert_dict_keys_equal(losses,
                               ['loss_alpha', 'loss_comp', 'loss_refine'])

    # test model forward in test mode
    with torch.no_grad():
        model = build_model(model_cfg, train_cfg=None, test_cfg=test_cfg)
        input_test = _demo_input_test((64, 64))
        prediction, metrics = model(**input_test, test_mode=True)
        assert isinstance(prediction, np.ndarray)
        assert_dict_keys_equal(metrics, ['SAD', 'MSE'])

        # test model forward in test mode with gpu
        if torch.cuda.is_available():
            model = build_model(model_cfg, train_cfg=None, test_cfg=test_cfg)
            model.cuda()
            input_test = _demo_input_test((64, 64), cuda=True)
            prediction, metrics = model(**input_test, test_mode=True)
            assert_dict_keys_equal(metrics, ['SAD', 'MSE'])

    # 2. test dim model without refiner
    model_cfg['refiner'] = None
    test_cfg['metrics'] = None

    # test model forward in train mode
    model = build_model(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)
    input_train = _demo_input_train((64, 64))
    losses = model(**input_train)
    assert_dict_keys_equal(losses, ['loss_alpha', 'loss_comp'])

    # test model forward in train mode with gpu
    if torch.cuda.is_available():
        model = build_model(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)
        model.cuda()
        input_train = _demo_input_train((64, 64), cuda=True)
        losses = model(**input_train)
        assert_dict_keys_equal(losses, ['loss_alpha', 'loss_comp'])

    # test model forward in test mode
    with torch.no_grad():
        model = build_model(model_cfg, train_cfg=None, test_cfg=test_cfg)
        input_test = _demo_input_test((64, 64))
        prediction, metrics = model(**input_test, test_mode=True)
        assert isinstance(prediction, np.ndarray)
        assert metrics is None

        # check test with gpu
        if torch.cuda.is_available():
            model = build_model(model_cfg, train_cfg=None, test_cfg=test_cfg)
            model.cuda()
            input_test = _demo_input_test((64, 64), cuda=True)
            prediction, metrics = model(**input_test, test_mode=True)
            assert isinstance(prediction, np.ndarray)
            assert metrics is None


def _demo_input_train(img_shape, batch_size=1, cuda=False):
    """
    Create a superset of inputs needed to run backbone.

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
        alpha=alpha,
        ori_merged=ori_merged,
        fg=fg,
        bg=bg)


def _demo_input_test(img_shape,
                     batch_size=1,
                     cuda=False,
                     test_trans='reshape'):
    """
    Create a superset of inputs needed to run backbone.

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
    img_meta = [
        dict(
            test_trans=test_trans,
            ori_alpha=ori_alpha,
            ori_trimap=ori_trimap,
            ori_shape=img_shape)
    ]

    return dict(merged=merged, trimap=trimap, img_meta=img_meta)
