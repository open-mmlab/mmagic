# Copyright (c) OpenMMLab. All rights reserved.
import copy

import numpy as np
import pytest
import torch

from mmedit.models import build_backbone
from mmedit.models.common import (ResidualBlockWithDropout,
                                  UnetSkipConnectionBlock)


def test_unet_skip_connection_block():
    _cfg = dict(
        outer_channels=1,
        inner_channels=1,
        in_channels=None,
        submodule=None,
        is_outermost=False,
        is_innermost=False,
        norm_cfg=dict(type='BN'),
        use_dropout=True)
    feature_shape = (1, 1, 8, 8)
    feature = _demo_inputs(feature_shape)
    input_shape = (1, 3, 8, 8)
    img = _demo_inputs(input_shape)

    # innermost
    cfg = copy.deepcopy(_cfg)
    cfg['is_innermost'] = True
    block = UnetSkipConnectionBlock(**cfg)
    # cpu
    output = block(feature)
    assert output.shape == (1, 2, 8, 8)
    # gpu
    if torch.cuda.is_available():
        block.cuda()
        output = block(feature.cuda())
        assert output.shape == (1, 2, 8, 8)
        block.cpu()

    # intermediate
    cfg = copy.deepcopy(_cfg)
    cfg['submodule'] = block
    block = UnetSkipConnectionBlock(**cfg)
    # cpu
    output = block(feature)
    assert output.shape == (1, 2, 8, 8)
    # gpu
    if torch.cuda.is_available():
        block.cuda()
        output = block(feature.cuda())
        assert output.shape == (1, 2, 8, 8)
        block.cpu()

    # outermost
    cfg = copy.deepcopy(_cfg)
    cfg['submodule'] = block
    cfg['is_outermost'] = True
    cfg['in_channels'] = 3
    cfg['outer_channels'] = 3
    block = UnetSkipConnectionBlock(**cfg)
    # cpu
    output = block(img)
    assert output.shape == (1, 3, 8, 8)
    # gpu
    if torch.cuda.is_available():
        block.cuda()
        output = block(img.cuda())
        assert output.shape == (1, 3, 8, 8)
        block.cpu()

    # test cannot be both innermost and outermost
    cfg = copy.deepcopy(_cfg)
    cfg['is_innermost'] = True
    cfg['is_outermost'] = True
    with pytest.raises(AssertionError):
        _ = UnetSkipConnectionBlock(**cfg)

    # test norm_cfg assertions
    bad_cfg = copy.deepcopy(_cfg)
    bad_cfg['is_innermost'] = True
    bad_cfg['norm_cfg'] = None
    with pytest.raises(AssertionError):
        _ = UnetSkipConnectionBlock(**bad_cfg)
    bad_cfg['norm_cfg'] = dict(tp='BN')
    with pytest.raises(AssertionError):
        _ = UnetSkipConnectionBlock(**bad_cfg)


def test_unet_generator():
    # color to color
    cfg = dict(
        type='UnetGenerator',
        in_channels=3,
        out_channels=3,
        num_down=8,
        base_channels=64,
        norm_cfg=dict(type='BN'),
        use_dropout=True,
        init_cfg=dict(type='normal', gain=0.02))
    net = build_backbone(cfg)
    net.init_weights(pretrained=None)
    # cpu
    input_shape = (1, 3, 256, 256)
    img = _demo_inputs(input_shape)
    output = net(img)
    assert output.shape == (1, 3, 256, 256)
    # gpu
    if torch.cuda.is_available():
        net = net.cuda()
        output = net(img.cuda())
        assert output.shape == (1, 3, 256, 256)

    # gray to color
    cfg = dict(
        type='UnetGenerator',
        in_channels=1,
        out_channels=3,
        num_down=8,
        base_channels=64,
        norm_cfg=dict(type='BN'),
        use_dropout=True,
        init_cfg=dict(type='normal', gain=0.02))
    net = build_backbone(cfg)
    net.init_weights(pretrained=None)
    # cpu
    input_shape = (1, 1, 256, 256)
    img = _demo_inputs(input_shape)
    output = net(img)
    assert output.shape == (1, 3, 256, 256)
    # gpu
    if torch.cuda.is_available():
        net = net.cuda()
        output = net(img.cuda())
        assert output.shape == (1, 3, 256, 256)

    # color to gray
    cfg = dict(
        type='UnetGenerator',
        in_channels=3,
        out_channels=1,
        num_down=8,
        base_channels=64,
        norm_cfg=dict(type='BN'),
        use_dropout=True,
        init_cfg=dict(type='normal', gain=0.02))
    net = build_backbone(cfg)
    net.init_weights(pretrained=None)
    # cpu
    input_shape = (1, 3, 256, 256)
    img = _demo_inputs(input_shape)
    output = net(img)
    assert output.shape == (1, 1, 256, 256)
    # gpu
    if torch.cuda.is_available():
        net = net.cuda()
        output = net(img.cuda())
        assert output.shape == (1, 1, 256, 256)

    # pretrained should be str or None
    with pytest.raises(TypeError):
        net.init_weights(pretrained=[1])

    # test norm_cfg assertions
    bad_cfg = copy.deepcopy(cfg)
    bad_cfg['norm_cfg'] = None
    with pytest.raises(AssertionError):
        _ = build_backbone(bad_cfg)
    bad_cfg['norm_cfg'] = dict(tp='BN')
    with pytest.raises(AssertionError):
        _ = build_backbone(bad_cfg)


def test_residual_block_with_dropout():
    _cfg = dict(
        channels=3,
        padding_mode='reflect',
        norm_cfg=dict(type='BN'),
        use_dropout=True)
    feature_shape = (1, 3, 32, 32)
    feature = _demo_inputs(feature_shape)
    # reflect padding, BN, use_dropout=True
    block = ResidualBlockWithDropout(**_cfg)
    # cpu
    output = block(feature)
    assert output.shape == (1, 3, 32, 32)
    # gpu
    if torch.cuda.is_available():
        block = block.cuda()
        output = block(feature.cuda())
        assert output.shape == (1, 3, 32, 32)

    # test other padding types
    # replicate padding
    cfg = copy.deepcopy(_cfg)
    cfg['padding_mode'] = 'replicate'
    block = ResidualBlockWithDropout(**cfg)
    # cpu
    output = block(feature)
    assert output.shape == (1, 3, 32, 32)
    # gpu
    if torch.cuda.is_available():
        block = block.cuda()
        output = block(feature.cuda())
        assert output.shape == (1, 3, 32, 32)
    # zero padding
    cfg = copy.deepcopy(_cfg)
    cfg['padding_mode'] = 'zeros'
    block = ResidualBlockWithDropout(**cfg)
    # cpu
    output = block(feature)
    assert output.shape == (1, 3, 32, 32)
    # gpu
    if torch.cuda.is_available():
        block = block.cuda()
        output = block(feature.cuda())
        assert output.shape == (1, 3, 32, 32)
    # not implemented padding
    cfg = copy.deepcopy(_cfg)
    cfg['padding_mode'] = 'abc'
    with pytest.raises(KeyError):
        block = ResidualBlockWithDropout(**cfg)

    # test other norm
    cfg = copy.deepcopy(_cfg)
    cfg['norm_cfg'] = dict(type='IN')
    block = ResidualBlockWithDropout(**cfg)
    # cpu
    output = block(feature)
    assert output.shape == (1, 3, 32, 32)
    # gpu
    if torch.cuda.is_available():
        block = block.cuda()
        output = block(feature.cuda())
        assert output.shape == (1, 3, 32, 32)

    # test use_dropout=False
    cfg = copy.deepcopy(_cfg)
    cfg['use_dropout'] = False
    block = ResidualBlockWithDropout(**cfg)
    # cpu
    output = block(feature)
    assert output.shape == (1, 3, 32, 32)
    # gpu
    if torch.cuda.is_available():
        block = block.cuda()
        output = block(feature.cuda())
        assert output.shape == (1, 3, 32, 32)

    # test norm_cfg assertions
    bad_cfg = copy.deepcopy(_cfg)
    bad_cfg['norm_cfg'] = None
    with pytest.raises(AssertionError):
        _ = ResidualBlockWithDropout(**bad_cfg)
    bad_cfg['norm_cfg'] = dict(tp='BN')
    with pytest.raises(AssertionError):
        _ = ResidualBlockWithDropout(**bad_cfg)


def test_resnet_generator():
    # color to color
    cfg = dict(
        type='ResnetGenerator',
        in_channels=3,
        out_channels=3,
        base_channels=64,
        norm_cfg=dict(type='IN'),
        use_dropout=False,
        num_blocks=9,
        padding_mode='reflect',
        init_cfg=dict(type='normal', gain=0.02))
    net = build_backbone(cfg)
    net.init_weights(pretrained=None)
    # cpu
    input_shape = (1, 3, 256, 256)
    img = _demo_inputs(input_shape)
    output = net(img)
    assert output.shape == (1, 3, 256, 256)
    # gpu
    if torch.cuda.is_available():
        net = net.cuda()
        output = net(img.cuda())
        assert output.shape == (1, 3, 256, 256)

    # gray to color
    cfg = dict(
        type='ResnetGenerator',
        in_channels=1,
        out_channels=3,
        base_channels=64,
        norm_cfg=dict(type='IN'),
        use_dropout=False,
        num_blocks=9,
        padding_mode='reflect',
        init_cfg=dict(type='normal', gain=0.02))
    net = build_backbone(cfg)
    net.init_weights(pretrained=None)
    # cpu
    input_shape = (1, 1, 256, 256)
    img = _demo_inputs(input_shape)
    output = net(img)
    assert output.shape == (1, 3, 256, 256)
    # gpu
    if torch.cuda.is_available():
        net = net.cuda()
        output = net(img.cuda())
        assert output.shape == (1, 3, 256, 256)

    # color to gray
    cfg = dict(
        type='ResnetGenerator',
        in_channels=3,
        out_channels=1,
        base_channels=64,
        norm_cfg=dict(type='IN'),
        use_dropout=False,
        num_blocks=9,
        padding_mode='reflect',
        init_cfg=dict(type='normal', gain=0.02))
    net = build_backbone(cfg)
    net.init_weights(pretrained=None)
    # cpu
    input_shape = (1, 3, 256, 256)
    img = _demo_inputs(input_shape)
    output = net(img)
    assert output.shape == (1, 1, 256, 256)
    # gpu
    if torch.cuda.is_available():
        net = net.cuda()
        output = net(img.cuda())
        assert output.shape == (1, 1, 256, 256)

    # test num_blocks non-negative
    bad_cfg = copy.deepcopy(cfg)
    bad_cfg['num_blocks'] = -1
    with pytest.raises(AssertionError):
        net = build_backbone(bad_cfg)

    # pretrained should be str or None
    with pytest.raises(TypeError):
        net.init_weights(pretrained=[1])

    # test norm_cfg assertions
    bad_cfg = copy.deepcopy(cfg)
    bad_cfg['norm_cfg'] = None
    with pytest.raises(AssertionError):
        _ = build_backbone(bad_cfg)
    bad_cfg['norm_cfg'] = dict(tp='IN')
    with pytest.raises(AssertionError):
        _ = build_backbone(bad_cfg)


def _demo_inputs(input_shape=(1, 3, 64, 64)):
    """Create a superset of inputs needed to run backbone.

    Args:
        input_shape (tuple): input batch dimensions.
            Default: (1, 3, 64, 64).

    Returns:
        imgs: (Tensor): Images in FloatTensor with desired shapes.
    """
    imgs = np.random.random(input_shape)
    imgs = torch.FloatTensor(imgs)

    return imgs
