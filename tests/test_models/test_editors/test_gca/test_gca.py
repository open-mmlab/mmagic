# Copyright (c) OpenMMLab. All rights reserved.

import numpy as np
import torch
from mmengine.config import ConfigDict

from mmagic.datasets.transforms import PackInputs
from mmagic.registry import MODELS
from mmagic.structures import DataSample
from mmagic.utils import register_all_modules

register_all_modules()


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
    ori_alpha = np.random.random(ori_shape).astype(np.float32)
    ori_trimap = np.random.randint(256, size=ori_shape).astype(np.float32)

    inputs = torch.cat((merged, trimap), dim=1)
    if cuda:
        inputs = inputs.cuda()

    results = dict(
        ori_alpha=ori_alpha, ori_trimap=ori_trimap, ori_merged_shape=img_shape)

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


def test_gca():
    model_cfg = ConfigDict(
        type='GCA',
        data_preprocessor=dict(
            type='MattorPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            proc_trimap='as_is',
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
