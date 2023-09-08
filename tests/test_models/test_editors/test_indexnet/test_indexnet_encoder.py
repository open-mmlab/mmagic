# Copyright (c) OpenMMLab. All rights reserved.
try:
    import Iterable
except ImportError:
    from collections.abc import Iterable
import pytest
import torch

from mmagic.models import (DepthwiseIndexBlock, HolisticIndexBlock,
                           IndexNetEncoder)


def test_indexnet_encoder():
    """Test Indexnet encoder."""
    with pytest.raises(ValueError):
        # out_stride must be 16 or 32
        IndexNetEncoder(4, out_stride=8)
    with pytest.raises(NameError):
        # index_mode must be 'holistic', 'o2o' or 'm2o'
        IndexNetEncoder(4, index_mode='unknown_mode')

    # test indexnet encoder with default indexnet setting
    indexnet_encoder = IndexNetEncoder(
        4,
        out_stride=32,
        width_mult=1,
        index_mode='m2o',
        aspp=True,
        use_nonlinear=True,
        use_context=True)
    indexnet_encoder.init_weights()
    x = torch.rand(2, 4, 32, 32)
    outputs = indexnet_encoder(x)
    assert outputs['out'].shape == (2, 160, 1, 1)
    assert len(outputs['shortcuts']) == 7
    target_shapes = [(2, 32, 32, 32), (2, 16, 16, 16), (2, 24, 16, 16),
                     (2, 32, 8, 8), (2, 64, 4, 4), (2, 96, 2, 2),
                     (2, 160, 2, 2)]
    for shortcut, target_shape in zip(outputs['shortcuts'], target_shapes):
        assert shortcut.shape == target_shape
    assert len(outputs['dec_idx_feat_list']) == 7
    target_shapes = [(2, 32, 32, 32), None, (2, 24, 16, 16), (2, 32, 8, 8),
                     (2, 64, 4, 4), None, (2, 160, 2, 2)]
    for dec_idx_feat, target_shape in zip(outputs['dec_idx_feat_list'],
                                          target_shapes):
        if dec_idx_feat is not None:
            assert dec_idx_feat.shape == target_shape

    # test indexnet encoder with other config
    indexnet_encoder = IndexNetEncoder(
        4,
        out_stride=16,
        width_mult=2,
        index_mode='o2o',
        aspp=False,
        use_nonlinear=False,
        use_context=False)
    indexnet_encoder.init_weights()
    x = torch.rand(2, 4, 32, 32)
    outputs = indexnet_encoder(x)
    assert outputs['out'].shape == (2, 160, 2, 2)
    assert len(outputs['shortcuts']) == 7
    target_shapes = [(2, 64, 32, 32), (2, 32, 16, 16), (2, 48, 16, 16),
                     (2, 64, 8, 8), (2, 128, 4, 4), (2, 192, 2, 2),
                     (2, 320, 2, 2)]
    for shortcut, target_shape in zip(outputs['shortcuts'], target_shapes):
        assert shortcut.shape == target_shape
    assert len(outputs['dec_idx_feat_list']) == 7
    target_shapes = [(2, 64, 32, 32), None, (2, 48, 16, 16), (2, 64, 8, 8),
                     (2, 128, 4, 4), None, None]
    for dec_idx_feat, target_shape in zip(outputs['dec_idx_feat_list'],
                                          target_shapes):
        if dec_idx_feat is not None:
            assert dec_idx_feat.shape == target_shape

    # test indexnet encoder with holistic index block
    indexnet_encoder = IndexNetEncoder(
        4,
        out_stride=16,
        width_mult=2,
        index_mode='holistic',
        aspp=False,
        freeze_bn=True,
        use_nonlinear=False,
        use_context=False)
    indexnet_encoder.init_weights()
    x = torch.rand(2, 4, 32, 32)
    outputs = indexnet_encoder(x)
    assert outputs['out'].shape == (2, 160, 2, 2)
    assert len(outputs['shortcuts']) == 7
    target_shapes = [(2, 64, 32, 32), (2, 32, 16, 16), (2, 48, 16, 16),
                     (2, 64, 8, 8), (2, 128, 4, 4), (2, 192, 2, 2),
                     (2, 320, 2, 2)]
    for shortcut, target_shape in zip(outputs['shortcuts'], target_shapes):
        assert shortcut.shape == target_shape
    assert len(outputs['dec_idx_feat_list']) == 7
    target_shapes = [(2, 1, 32, 32), None, (2, 1, 16, 16), (2, 1, 8, 8),
                     (2, 1, 4, 4), None, None]
    for dec_idx_feat, target_shape in zip(outputs['dec_idx_feat_list'],
                                          target_shapes):
        if dec_idx_feat is not None:
            assert dec_idx_feat.shape == target_shape


def test_index_blocks():
    """Test index blocks for indexnet encoder."""
    # test holistic index block
    # test holistic index block without context and nonlinearty
    block = HolisticIndexBlock(128, use_context=False, use_nonlinear=False)
    assert not isinstance(block.index_block, Iterable)
    x = torch.rand(2, 128, 8, 8)
    enc_idx_feat, dec_idx_feat = block(x)
    assert enc_idx_feat.shape == (2, 1, 8, 8)
    assert dec_idx_feat.shape == (2, 1, 8, 8)

    # test holistic index block with context and nonlinearty
    block = HolisticIndexBlock(128, use_context=True, use_nonlinear=True)
    assert len(block.index_block) == 2  # nonlinear mode has two blocks
    x = torch.rand(2, 128, 8, 8)
    enc_idx_feat, dec_idx_feat = block(x)
    assert enc_idx_feat.shape == (2, 1, 8, 8)
    assert dec_idx_feat.shape == (2, 1, 8, 8)

    # test depthwise index block
    # test depthwise index block without context and nonlinearty in o2o mode
    block = DepthwiseIndexBlock(
        128, use_context=False, mode='oso', use_nonlinear=False)
    assert not isinstance(block.index_blocks[0], Iterable)
    x = torch.rand(2, 128, 8, 8)
    enc_idx_feat, dec_idx_feat = block(x)
    assert enc_idx_feat.shape == (2, 128, 8, 8)
    assert dec_idx_feat.shape == (2, 128, 8, 8)

    # test depthwise index block with context and nonlinearty in m2o mode
    block = DepthwiseIndexBlock(
        128, use_context=True, mode='m2o', use_nonlinear=True)
    assert len(block.index_blocks[0]) == 2  # nonlinear mode has two blocks
    x = torch.rand(2, 128, 8, 8)
    enc_idx_feat, dec_idx_feat = block(x)
    assert enc_idx_feat.shape == (2, 128, 8, 8)
    assert dec_idx_feat.shape == (2, 128, 8, 8)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
