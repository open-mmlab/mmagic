# Copyright (c) OpenMMLab. All rights reserved.
import platform

import numpy as np
import pytest
import torch
from mmengine.config import ConfigDict

from mmagic.datasets.transforms import PackInputs
from mmagic.models.editors import DIM
from mmagic.registry import MODELS
from mmagic.structures import DataSample
from mmagic.utils import register_all_modules

register_all_modules()


@pytest.mark.skipif(
    'win' in platform.system().lower() and 'cu' in torch.__version__,
    reason='skip on windows-cuda due to limited RAM.')
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

    inputs = torch.cat((merged, trimap), dim=1)
    data_samples = []
    for a, m, f, b in zip(alpha, ori_merged, fg, bg):
        ds = DataSample()

        ds.gt_alpha = a
        ds.gt_merged = m
        ds.gt_fg = f
        ds.gt_bg = b
        for k, v in meta.items():
            ds.set_field(name=k, value=v, field_type='metainfo', dtype=None)

        data_samples.append(ds)

    data_samples = DataSample.stack(data_samples)
    return inputs, data_samples


@pytest.mark.skipif(
    'win' in platform.system().lower() and 'cu' in torch.__version__,
    reason='skip on windows-cuda due to limited RAM.')
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

    results = {
        'ori_alpha': np.random.random(ori_shape).astype(np.float32),
        'ori_trimap': np.random.randint(256,
                                        size=ori_shape).astype(np.float32),
        'ori_merged_shape': img_shape,
    }
    packinputs = PackInputs()

    data_samples = []
    for _ in range(batch_size):
        ds = packinputs(results)['data_samples']
        if cuda:
            ds = ds.cuda()
        data_samples.append(ds)

    if cuda:
        inputs = inputs.cuda()
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


@pytest.mark.skipif(
    'win' in platform.system().lower() and 'cu' in torch.__version__,
    reason='skip on windows-cuda due to limited RAM.')
def test_dim_config():

    data_preprocessor = dict(
        type='MattorPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        proc_trimap='rescale_to_zero_one',
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


@pytest.mark.skipif(
    'win' in platform.system().lower() and 'cu' in torch.__version__,
    reason='skip on windows-cuda due to limited RAM.')
def test_dim():
    model_cfg = ConfigDict(
        type='DIM',
        data_preprocessor=dict(
            type='MattorPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            proc_trimap='rescale_to_zero_one',
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
        inputs, data_samples = _demo_input_test((48, 48))
        output_test = model(inputs, data_samples, mode='predict')
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


test_dim()


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
