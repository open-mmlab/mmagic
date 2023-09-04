# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from unittest.mock import patch

import mmcv
import numpy as np
import torch
from mmengine.config import ConfigDict

from mmagic.datasets.transforms import PackInputs
from mmagic.models.base_models import BaseMattor
from mmagic.models.editors import DIM
from mmagic.registry import MODELS
from mmagic.structures import DataSample
from mmagic.utils import register_all_modules

register_all_modules()


def _get_model_cfg(fname):
    """Grab configs necessary to create a model.

    These are deep copied to allow for safe modification of parameters without
    influencing other tests.
    """
    config_dpath = 'configs'
    config_fpath = osp.join(config_dpath, fname)
    if not osp.exists(config_dpath):
        raise Exception('Cannot find config path')
    config = mmcv.Config.fromfile(config_fpath)
    return config.model


def _demo_input_train(img_shape, batch_size=1, cuda=False, meta={}):
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
    inputs = torch.cat((merged, trimap), dim=1)
    if cuda:
        inputs = inputs.cuda()

    results = dict(
        alpha=np.random.random(
            (img_shape[0], img_shape[1], 1)).astype(np.float32),
        merged=np.random.random(
            (img_shape[0], img_shape[1], 3)).astype(np.float32),
        fg=np.random.random(
            (img_shape[0], img_shape[1], 3)).astype(np.float32),
        bg=np.random.random(
            (img_shape[0], img_shape[1], 3)).astype(np.float32))

    data_samples = []
    packinputs = PackInputs()
    for _ in range(batch_size):
        ds = packinputs(results)['data_samples']
        if cuda:
            ds = ds.cuda()
        data_samples.append(ds)

    data_samples = DataSample.stack(data_samples)
    return inputs, data_samples


def _demo_input_test(img_shape, batch_size=1, cuda=False, meta={}):
    """Create a superset of inputs needed to run backbone.

    Args:
        img_shape (tuple): shape of the input image.
        batch_size (int): batch size of the input batch.
        cuda (bool): whether transfer input into gpu.
        test_trans (str): what test transformation is used in data pipeline.
    """
    color_shape = (batch_size, 3, img_shape[0], img_shape[1])
    gray_shape = (batch_size, 1, img_shape[0], img_shape[1])
    ori_shape = (img_shape[0], img_shape[1], 1)

    merged = torch.from_numpy(np.random.random(color_shape).astype(np.float32))
    trimap = torch.from_numpy(
        np.random.randint(255, size=gray_shape).astype(np.float32))
    inputs = torch.cat((merged, trimap), dim=1)
    if cuda:
        inputs = inputs.cuda()

    results = dict(
        ori_alpha=np.random.random(ori_shape).astype(np.float32),
        ori_trimap=np.random.randint(256, size=ori_shape).astype(np.float32),
        ori_merged_shape=img_shape)

    data_samples = []
    packinputs = PackInputs()
    for _ in range(batch_size):
        ds = packinputs(results)['data_samples']
        if cuda:
            ds = ds.cuda()
        data_samples.append(ds)
    data_samples = DataSample.stack(data_samples)
    return inputs, data_samples


def assert_pred_alpha(predictions, batch_size):
    assert isinstance(predictions, list)
    assert isinstance(predictions[0], DataSample)
    pred_alpha = predictions[0].output.pred_alpha.data
    assert isinstance(pred_alpha, torch.Tensor)
    assert pred_alpha.dtype == torch.uint8
    assert pred_alpha.shape[-2:] == batch_size


def assert_dict_keys_equal(dictionary, target_keys):
    """Check if the keys of the dictionary is equal to the target key set."""
    assert isinstance(dictionary, dict)
    assert set(dictionary.keys()) == set(target_keys)


@patch.multiple(BaseMattor, __abstractmethods__=set())
def test_base_mattor():
    """Test resize and restore size."""

    data_preprocessor = dict(type='MattorPreprocessor')
    backbone = dict(
        type='SimpleEncoderDecoder',
        encoder=dict(type='VGG16', in_channels=4),
        decoder=dict(type='PlainDecoder'))
    test_cfgs = [
        dict(resize_method='pad', resize_mode='reflect', size_divisor=32),
        dict(resize_method='interp', resize_mode='bilinear', size_divisor=32),
    ]
    out_shapes = (64, 64), (32, 32)

    inputs = torch.rand((1, 4, 48, 48))
    for test_cfg, out_shape in zip(test_cfgs, out_shapes):
        mattor = BaseMattor(
            data_preprocessor=data_preprocessor,
            backbone=backbone,
            test_cfg=test_cfg,
        )

        out = mattor.resize_inputs(inputs)
        assert out.shape[-2:] == out_shape
        out = mattor.restore_size(out[0],
                                  ConfigDict(ori_merged_shape=(48, 48)))
        assert out.shape[-2:] == (48, 48)


def test_dim_config():

    data_preprocessor = dict(
        type='MattorPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        # bgr_to_rgb=True,
        # proc_inputs='normalize',
        proc_trimap='rescale_to_zero_one',
        # proc_gt='rescale_to_zero_one',
    )
    backbone = dict(
        type='SimpleEncoderDecoder',
        encoder=dict(type='VGG16', in_channels=4),
        decoder=dict(type='PlainDecoder'))
    refiner = dict(type='PlainRefiner')
    train_cfg = dict(train_backbone=True, train_refiner=True)
    test_cfg = dict(refine=True)
    loss_alpha = dict(type='L1Loss')

    # build mattor without refiner
    mattor = DIM(
        data_preprocessor,
        backbone,
        refiner=None,
        loss_alpha=loss_alpha,
        train_cfg=train_cfg,
        test_cfg=test_cfg.copy())
    assert not mattor.with_refiner
    assert not mattor.test_cfg.refine

    # only train the refiner, this will freeze the backbone
    mattor = DIM(
        data_preprocessor,
        backbone,
        refiner,
        loss_alpha=loss_alpha,
        train_cfg=dict(train_backbone=False, train_refiner=True),
        test_cfg=test_cfg.copy())
    assert not mattor.train_cfg.train_backbone
    assert mattor.train_cfg.train_refiner
    assert mattor.test_cfg.refine

    # only train the backbone while the refiner is used for inference but not
    # trained, this behavior is allowed currently but will cause a warning.
    mattor = DIM(
        data_preprocessor,
        backbone,
        refiner,
        loss_alpha=loss_alpha,
        train_cfg=dict(train_backbone=True, train_refiner=False),
        test_cfg=test_cfg.copy())
    assert mattor.train_cfg.train_backbone
    assert not mattor.train_cfg.train_refiner
    assert mattor.test_cfg.refine


def test_dim():
    model_cfg = ConfigDict(
        type='DIM',
        data_preprocessor=dict(
            type='MattorPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            # bgr_to_rgb=True,
            # proc_inputs='normalize',
            proc_trimap='rescale_to_zero_one',
            # proc_gt='rescale_to_zero_one',
        ),
        backbone=dict(
            type='SimpleEncoderDecoder',
            encoder=dict(type='VGG16', in_channels=4),
            decoder=dict(type='PlainDecoder')),
        refiner=dict(type='PlainRefiner'),
        loss_alpha=dict(type='CharbonnierLoss', loss_weight=0.5),
        loss_comp=dict(type='CharbonnierCompLoss', loss_weight=0.5),
        loss_refine=dict(type='CharbonnierLoss'),
        train_cfg=dict(train_backbone=True, train_refiner=True),
        test_cfg=dict(
            refine=False,
            resize_method='pad',
            resize_mode='reflect',
            size_divisor=32,
        ),
    )

    # 1. test dim model with refiner
    model_cfg.train_cfg.train_refiner = True
    model_cfg.test_cfg.refine = True

    # test model forward in train mode
    model = MODELS.build(model_cfg)
    input_train = _demo_input_train((64, 64))
    output_train = model(*input_train, mode='loss')
    assert_dict_keys_equal(output_train,
                           ['loss_alpha', 'loss_comp', 'loss_refine'])

    # test model forward in train mode with gpu
    if torch.cuda.is_available():
        model = MODELS.build(model_cfg)
        model.cuda()
        input_train = _demo_input_train((64, 64), cuda=True)
        output_train = model(*input_train, mode='loss')
        assert_dict_keys_equal(output_train,
                               ['loss_alpha', 'loss_comp', 'loss_refine'])

    # test model forward in test mode
    with torch.no_grad():
        model = MODELS.build(model_cfg)
        input_test = _demo_input_test((48, 48))
        output_test = model(*input_test, mode='predict')
        assert isinstance(output_test, list)
        assert isinstance(output_test[0], DataSample)
        pred_alpha = output_test[0].output.pred_alpha.data
        assert isinstance(pred_alpha, torch.Tensor)
        assert pred_alpha.dtype == torch.uint8
        assert pred_alpha.shape[-2:] == (48, 48)

        # test model forward in test mode with gpu
        if torch.cuda.is_available():
            model = MODELS.build(model_cfg)
            model.cuda()
            input_test = _demo_input_test((48, 48), cuda=True)
            output_test = model(*input_test, mode='predict')
            assert_pred_alpha(output_test, (48, 48))

    # 2. test dim model without refiner
    model_cfg.refiner = None
    model_cfg.test_cfg.refine = True

    # test model forward in train mode
    model = MODELS.build(model_cfg)
    input_train = _demo_input_train((64, 64))
    output_train = model(*input_train, mode='loss')
    assert_dict_keys_equal(output_train, ['loss_alpha', 'loss_comp'])

    # test model forward in train mode with gpu
    if torch.cuda.is_available():
        model = MODELS.build(model_cfg)
        model.cuda()
        input_train = _demo_input_train((64, 64), cuda=True)
        output_train = model(*input_train, mode='loss')
        assert_dict_keys_equal(output_train, ['loss_alpha', 'loss_comp'])

    # test model forward in test mode
    with torch.no_grad():
        model = MODELS.build(model_cfg)
        input_test = _demo_input_test((48, 48))
        output_test = model(*input_test, mode='predict')
        assert_pred_alpha(output_test, (48, 48))

        # check test with gpu
        if torch.cuda.is_available():
            model = MODELS.build(model_cfg)
            model.cuda()
            input_test = _demo_input_test((48, 48), cuda=True)
            output_test = model(*input_test, mode='predict')

    # test forward_raw
    model.cpu().eval()
    inputs = torch.ones((1, 4, 32, 32))
    model.forward(inputs)


def test_indexnet():
    model_cfg = ConfigDict(
        type='IndexNet',
        data_preprocessor=dict(
            type='MattorPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            # bgr_to_rgb=True,
            # proc_inputs='normalize',
            proc_trimap='rescale_to_zero_one',
            # proc_gt='rescale_to_zero_one',
        ),
        backbone=dict(
            type='SimpleEncoderDecoder',
            encoder=dict(
                type='IndexNetEncoder',
                in_channels=4,
                freeze_bn=True,
            ),
            decoder=dict(type='IndexNetDecoder')),
        loss_alpha=dict(
            type='CharbonnierLoss', loss_weight=0.5, sample_wise=True),
        loss_comp=dict(
            type='CharbonnierCompLoss', loss_weight=1.5, sample_wise=True),
        test_cfg=dict(
            resize_method='interp',
            resize_mode='bicubic',
            size_divisor=32,
        ),
    )

    model_cfg.backbone.encoder.init_cfg = None

    # test indexnet inference
    with torch.no_grad():
        indexnet = MODELS.build(model_cfg)
        indexnet.eval()
        input_test = _demo_input_test((48, 48))
        output_test = indexnet(*input_test, mode='predict')
        assert_pred_alpha(output_test, (48, 48))

        # test inference with gpu
        if torch.cuda.is_available():
            indexnet = MODELS.build(model_cfg).cuda()
            indexnet.eval()
            input_test = _demo_input_test((48, 48), cuda=True)
            output_test = indexnet(*input_test, mode='predict')
        assert_pred_alpha(output_test, (48, 48))

    # test forward train though we do not guarantee the training for present
    model_cfg.loss_alpha = None
    model_cfg.loss_comp = dict(type='L1CompositionLoss')
    indexnet = MODELS.build(model_cfg)
    input_train = _demo_input_train((64, 64), batch_size=2)
    output_train = indexnet(*input_train, mode='loss')
    # assert output_train['num_samples'] == 2
    assert_dict_keys_equal(output_train, ['loss_comp'])

    if torch.cuda.is_available():
        model_cfg.loss_alpha = dict(type='L1Loss')
        model_cfg.loss_comp = None
        indexnet = MODELS.build(model_cfg).cuda()
        input_train = _demo_input_train((64, 64), batch_size=2, cuda=True)
        output_train = indexnet(*input_train, mode='loss')
        # assert output_train['num_samples'] == 2
        assert_dict_keys_equal(output_train, ['loss_alpha'])

    # test forward
    indexnet.cpu().eval()
    inputs = torch.ones((1, 4, 32, 32))
    indexnet.forward(inputs)


def test_gca():
    model_cfg = ConfigDict(
        type='GCA',
        data_preprocessor=dict(
            type='MattorPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            # bgr_to_rgb=True,
            # proc_inputs='normalize',
            proc_trimap='as_is',
            # proc_gt='rescale_to_zero_one',
        ),
        backbone=dict(
            type='SimpleEncoderDecoder',
            encoder=dict(
                type='ResGCAEncoder',
                block='BasicBlock',
                layers=[3, 4, 4, 2],
                in_channels=6,
                with_spectral_norm=True,
            ),
            decoder=dict(
                type='ResGCADecoder',
                block='BasicBlockDec',
                layers=[2, 3, 3, 2],
                with_spectral_norm=True)),
        loss_alpha=dict(type='L1Loss'),
        test_cfg=dict(
            resize_method='pad',
            resize_mode='reflect',
            size_divisor=32,
        ))

    # test model forward in train mode
    model = MODELS.build(model_cfg)
    meta = {'format_trimap_to_onehot': True}
    inputs, data_samples = _demo_input_train((64, 64), batch_size=2, meta=meta)
    inputs6 = torch.cat((inputs, inputs[:, 3:, :, :], inputs[:, 3:, :, :]),
                        dim=1)
    outputs = model(inputs6, data_samples, mode='loss')
    assert_dict_keys_equal(outputs, ['loss'])

    if torch.cuda.is_available():
        model = MODELS.build(model_cfg)
        model.cuda()
        inputs, data_samples = _demo_input_train((64, 64),
                                                 batch_size=2,
                                                 cuda=True,
                                                 meta=meta)
        inputs6 = torch.cat((inputs, inputs[:, 3:, :, :], inputs[:, 3:, :, :]),
                            dim=1)
        outputs = model(inputs6, data_samples, mode='loss')
        assert_dict_keys_equal(outputs, ['loss'])

    # test model forward in test mode
    with torch.no_grad():
        model_cfg.backbone.encoder.in_channels = 4
        model = MODELS.build(model_cfg)
        inputs = _demo_input_test((48, 48), meta=meta)
        outputs = model(*inputs, mode='predict')
        assert_pred_alpha(outputs, (48, 48))

        if torch.cuda.is_available():
            model = MODELS.build(model_cfg)
            model.cuda()
            inputs = _demo_input_test((48, 48), cuda=True, meta=meta)
            outputs = model(*inputs, mode='predict')
            assert_pred_alpha(outputs, (48, 48))

    # test forward_dummy
    model.cpu().eval()
    inputs = torch.ones((1, 4, 32, 32))
    model.forward(inputs)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
