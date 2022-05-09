# Copyright (c) OpenMMLab. All rights reserved.
import os

import mmcv
import numpy as np
import pytest
import torch
from packaging import version

from mmedit.models import build_model


@pytest.mark.skipif(torch.__version__ == 'parrots', reason='skip parrots.')
@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse('1.4.0'),
    reason='skip if torch=1.3.x')
def test_restorer_wrapper():
    try:
        import onnxruntime as ort

        from mmedit.core.export.wrappers import (ONNXRuntimeEditing,
                                                 ONNXRuntimeRestorer)
    except ImportError:
        pytest.skip('ONNXRuntime is not available.')

    onnx_path = 'tmp.onnx'
    scale = 4
    train_cfg = None
    test_cfg = None
    cfg = dict(
        model=dict(
            type='BasicRestorer',
            generator=dict(
                type='SRCNN',
                channels=(3, 4, 2, 3),
                kernel_sizes=(9, 1, 5),
                upscale_factor=scale),
            pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean')),
        train_cfg=train_cfg,
        test_cfg=test_cfg)
    cfg = mmcv.Config(cfg)

    pytorch_model = build_model(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    # prepare data
    inputs = torch.rand(1, 3, 2, 2)
    targets = torch.rand(1, 3, 8, 8)
    data_batch = {'lq': inputs, 'gt': targets}

    pytorch_model.forward = pytorch_model.forward_dummy
    with torch.no_grad():
        torch.onnx.export(
            pytorch_model,
            inputs,
            onnx_path,
            input_names=['input'],
            output_names=['output'],
            export_params=True,
            keep_initializers_as_inputs=False,
            verbose=False,
            opset_version=11)

    wrap_model = ONNXRuntimeEditing(onnx_path, cfg, 0)
    # os.remove(onnx_path)
    assert isinstance(wrap_model.wrapper, ONNXRuntimeRestorer)

    if ort.get_device() == 'GPU':
        data_batch = {'lq': inputs.cuda(), 'gt': targets.cuda()}

    with torch.no_grad():
        outputs = wrap_model(**data_batch, test_mode=True)

    assert isinstance(outputs, dict)
    assert 'output' in outputs
    output = outputs['output']
    assert isinstance(output, torch.Tensor)
    assert output.shape == targets.shape


@pytest.mark.skipif(torch.__version__ == 'parrots', reason='skip parrots.')
@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse('1.4.0'),
    reason='skip if torch=1.3.x')
def test_mattor_wrapper():
    try:
        import onnxruntime as ort

        from mmedit.core.export.wrappers import (ONNXRuntimeEditing,
                                                 ONNXRuntimeMattor)
    except ImportError:
        pytest.skip('ONNXRuntime is not available.')
    onnx_path = 'tmp.onnx'
    train_cfg = None
    test_cfg = dict(refine=False, metrics=['SAD', 'MSE', 'GRAD', 'CONN'])
    cfg = dict(
        model=dict(
            type='DIM',
            backbone=dict(
                type='SimpleEncoderDecoder',
                encoder=dict(type='VGG16', in_channels=4),
                decoder=dict(type='PlainDecoder')),
            pretrained='open-mmlab://mmedit/vgg16',
            loss_alpha=dict(type='CharbonnierLoss', loss_weight=0.5),
            loss_comp=dict(type='CharbonnierCompLoss', loss_weight=0.5)),
        train_cfg=train_cfg,
        test_cfg=test_cfg)
    cfg = mmcv.Config(cfg)

    pytorch_model = build_model(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    img_shape = (32, 32)
    merged = torch.rand(1, 3, img_shape[1], img_shape[0])
    trimap = torch.rand(1, 1, img_shape[1], img_shape[0])
    data_batch = {'merged': merged, 'trimap': trimap}
    inputs = torch.cat([merged, trimap], dim=1)

    pytorch_model.forward = pytorch_model.forward_dummy
    with torch.no_grad():
        torch.onnx.export(
            pytorch_model,
            inputs,
            onnx_path,
            input_names=['input'],
            output_names=['output'],
            export_params=True,
            keep_initializers_as_inputs=False,
            verbose=False,
            opset_version=11)

    wrap_model = ONNXRuntimeEditing(onnx_path, cfg, 0)
    os.remove(onnx_path)
    assert isinstance(wrap_model.wrapper, ONNXRuntimeMattor)

    if ort.get_device() == 'GPU':
        merged = merged.cuda()
        trimap = trimap.cuda()
        data_batch = {'merged': merged, 'trimap': trimap}

    ori_alpha = np.random.random(img_shape).astype(np.float32)
    ori_trimap = np.random.randint(256, size=img_shape).astype(np.float32)
    data_batch['meta'] = [
        dict(
            ori_alpha=ori_alpha,
            ori_trimap=ori_trimap,
            merged_ori_shape=img_shape)
    ]

    with torch.no_grad():
        outputs = wrap_model(**data_batch, test_mode=True)

    assert isinstance(outputs, dict)
    assert 'pred_alpha' in outputs
    pred_alpha = outputs['pred_alpha']
    assert isinstance(pred_alpha, np.ndarray)
    assert pred_alpha.shape == img_shape
