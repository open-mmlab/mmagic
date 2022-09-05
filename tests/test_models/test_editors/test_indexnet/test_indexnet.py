# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmengine.config import ConfigDict

from mmedit.models.editors import IndexedUpsample
from mmedit.registry import MODELS
from mmedit.structures import EditDataSample, PixelData
from mmedit.utils import register_all_modules


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
        ds = EditDataSample()

        ds.gt_alpha = PixelData(data=a)
        ds.gt_merged = PixelData(data=m)
        ds.gt_fg = PixelData(data=f)
        ds.gt_bg = PixelData(data=b)
        for k, v in meta.items():
            ds.set_field(name=k, value=v, field_type='metainfo', dtype=None)

        data_samples.append(ds)

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
    if cuda:
        merged = merged.cuda()
        trimap = trimap.cuda()
    meta = dict(
        ori_alpha=ori_alpha,
        ori_trimap=ori_trimap,
        ori_merged_shape=img_shape,
        **meta)

    inputs = torch.cat((merged, trimap), dim=1)
    data_samples = []
    for _ in range(batch_size):
        ds = EditDataSample(metainfo=meta)
        data_samples.append(ds)

    return inputs, data_samples


def assert_pred_alpha(predictions, batch_size):
    assert isinstance(predictions, list)
    assert isinstance(predictions[0], EditDataSample)
    pred_alpha = predictions[0].output.pred_alpha.data
    assert isinstance(pred_alpha, torch.Tensor)
    assert pred_alpha.dtype == torch.uint8
    assert pred_alpha.shape[-2:] == batch_size


def assert_dict_keys_equal(dictionary, target_keys):
    """Check if the keys of the dictionary is equal to the target key set."""
    assert isinstance(dictionary, dict)
    assert set(dictionary.keys()) == set(target_keys)


def assert_tensor_with_shape(tensor, shape):
    """"Check if the shape of the tensor is equal to the target shape."""
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == shape


def _demo_inputs(input_shape=(1, 4, 64, 64)):
    """Create a superset of inputs needed to run encoder.

    Args:
        input_shape (tuple): input batch dimensions.
            Default: (1, 4, 64, 64).
    """
    img = np.random.random(input_shape).astype(np.float32)
    img = torch.from_numpy(img)

    return img


def test_indexnet():
    register_all_modules()
    model_cfg = ConfigDict(
        type='IndexNet',
        data_preprocessor=dict(
            type='MattorPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True,
            proc_inputs='normalize',
            proc_trimap='rescale_to_zero_one',
            proc_gt='rescale_to_zero_one',
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


def test_indexed_upsample():
    """Test indexed upsample module for indexnet decoder."""
    indexed_upsample = IndexedUpsample(12, 12)

    # test indexed_upsample without dec_idx_feat (no upsample)
    x = torch.rand(2, 6, 32, 32)
    shortcut = torch.rand(2, 6, 32, 32)
    output = indexed_upsample(x, shortcut)
    assert_tensor_with_shape(output, (2, 12, 32, 32))

    # test indexed_upsample without dec_idx_feat (with upsample)
    x = torch.rand(2, 6, 32, 32)
    dec_idx_feat = torch.rand(2, 6, 64, 64)
    shortcut = torch.rand(2, 6, 64, 64)
    output = indexed_upsample(x, shortcut, dec_idx_feat)
    assert_tensor_with_shape(output, (2, 12, 64, 64))
