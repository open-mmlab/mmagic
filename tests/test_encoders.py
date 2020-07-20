from collections.abc import Iterable

import numpy as np
import pytest
import torch
from mmcv.utils.parrots_wrapper import _BatchNorm

from mmedit.models.backbones import (VGG16, DepthwiseIndexBlock,
                                     HolisticIndexBlock, IndexNetEncoder,
                                     ResGCAEncoder, ResNetEnc, ResShortcutEnc)


def check_norm_state(modules, train_state):
    """Check if norm layer is in correct train state."""
    for mod in modules:
        if isinstance(mod, _BatchNorm):
            if mod.training != train_state:
                return False
    return True


def assert_tensor_with_shape(tensor, shape):
    """"Check if the shape of the tensor is equal to the target shape."""
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == shape


def assert_mid_feat_shape(mid_feat, target_shape):
    assert len(mid_feat) == 5
    for i in range(5):
        assert_tensor_with_shape(mid_feat[i], torch.Size(target_shape[i]))


def test_vgg16_encoder():
    """Test VGG16 encoder."""
    target_shape = [(2, 64, 32, 32), (2, 128, 16, 16), (2, 256, 8, 8),
                    (2, 512, 4, 4), (2, 512, 2, 2)]

    model = VGG16(4)
    model.init_weights()
    model.train()
    img = _demo_inputs()
    outputs = model(img)
    assert_tensor_with_shape(outputs['out'], (2, 512, 2, 2))
    assert_tensor_with_shape(outputs['max_idx_1'], target_shape[0])
    assert_tensor_with_shape(outputs['max_idx_2'], target_shape[1])
    assert_tensor_with_shape(outputs['max_idx_3'], target_shape[2])
    assert_tensor_with_shape(outputs['max_idx_4'], target_shape[3])
    assert_tensor_with_shape(outputs['max_idx_5'], target_shape[4])

    model = VGG16(4, batch_norm=True)
    model.init_weights()
    model.train()
    img = _demo_inputs()
    outputs = model(img)
    assert_tensor_with_shape(outputs['out'], (2, 512, 2, 2))
    assert_tensor_with_shape(outputs['max_idx_1'], target_shape[0])
    assert_tensor_with_shape(outputs['max_idx_2'], target_shape[1])
    assert_tensor_with_shape(outputs['max_idx_3'], target_shape[2])
    assert_tensor_with_shape(outputs['max_idx_4'], target_shape[3])
    assert_tensor_with_shape(outputs['max_idx_5'], target_shape[4])

    model = VGG16(4, aspp=True, dilations=[6, 12, 18])
    model.init_weights()
    model.train()
    img = _demo_inputs()
    outputs = model(img)
    assert_tensor_with_shape(outputs['out'], (2, 256, 2, 2))
    assert_tensor_with_shape(outputs['max_idx_1'], target_shape[0])
    assert_tensor_with_shape(outputs['max_idx_2'], target_shape[1])
    assert_tensor_with_shape(outputs['max_idx_3'], target_shape[2])
    assert_tensor_with_shape(outputs['max_idx_4'], target_shape[3])
    assert_tensor_with_shape(outputs['max_idx_5'], target_shape[4])
    assert check_norm_state(model.modules(), True)

    # test forward with gpu
    if torch.cuda.is_available():
        model = VGG16(4)
        model.init_weights()
        model.train()
        model.cuda()
        img = _demo_inputs().cuda()
        outputs = model(img)
        assert_tensor_with_shape(outputs['out'], (2, 512, 2, 2))
        assert_tensor_with_shape(outputs['max_idx_1'], target_shape[0])
        assert_tensor_with_shape(outputs['max_idx_2'], target_shape[1])
        assert_tensor_with_shape(outputs['max_idx_3'], target_shape[2])
        assert_tensor_with_shape(outputs['max_idx_4'], target_shape[3])
        assert_tensor_with_shape(outputs['max_idx_5'], target_shape[4])

        model = VGG16(4, batch_norm=True)
        model.init_weights()
        model.train()
        model.cuda()
        img = _demo_inputs().cuda()
        outputs = model(img)
        assert_tensor_with_shape(outputs['out'], (2, 512, 2, 2))
        assert_tensor_with_shape(outputs['max_idx_1'], target_shape[0])
        assert_tensor_with_shape(outputs['max_idx_2'], target_shape[1])
        assert_tensor_with_shape(outputs['max_idx_3'], target_shape[2])
        assert_tensor_with_shape(outputs['max_idx_4'], target_shape[3])
        assert_tensor_with_shape(outputs['max_idx_5'], target_shape[4])

        model = VGG16(4, aspp=True, dilations=[6, 12, 18])
        model.init_weights()
        model.train()
        model.cuda()
        img = _demo_inputs().cuda()
        outputs = model(img)
        assert_tensor_with_shape(outputs['out'], (2, 256, 2, 2))
        assert_tensor_with_shape(outputs['max_idx_1'], target_shape[0])
        assert_tensor_with_shape(outputs['max_idx_2'], target_shape[1])
        assert_tensor_with_shape(outputs['max_idx_3'], target_shape[2])
        assert_tensor_with_shape(outputs['max_idx_4'], target_shape[3])
        assert_tensor_with_shape(outputs['max_idx_5'], target_shape[4])
        assert check_norm_state(model.modules(), True)


def test_resnet_encoder():
    """Test resnet encoder."""
    with pytest.raises(NotImplementedError):
        ResNetEnc('UnknownBlock', [3, 4, 4, 2], 3)

    with pytest.raises(TypeError):
        model = ResNetEnc('BasicBlock', [3, 4, 4, 2], 3)
        model.init_weights(list())

    model = ResNetEnc('BasicBlock', [3, 4, 4, 2], 4, with_spectral_norm=True)
    assert hasattr(model.conv1.conv, 'weight_orig')
    model.init_weights()
    model.train()
    # trimap has 1 channels
    img = _demo_inputs((2, 4, 64, 64))
    feat = model(img)
    assert_tensor_with_shape(feat, torch.Size([2, 512, 2, 2]))

    # test resnet encoder with late downsample
    model = ResNetEnc('BasicBlock', [3, 4, 4, 2], 6, late_downsample=True)
    model.init_weights()
    model.train()
    # both image and trimap has 3 channels
    img = _demo_inputs((2, 6, 64, 64))
    feat = model(img)
    assert_tensor_with_shape(feat, torch.Size([2, 512, 2, 2]))

    if torch.cuda.is_available():
        # repeat above code again
        model = ResNetEnc(
            'BasicBlock', [3, 4, 4, 2], 4, with_spectral_norm=True)
        assert hasattr(model.conv1.conv, 'weight_orig')
        model.init_weights()
        model.train()
        model.cuda()
        # trimap has 1 channels
        img = _demo_inputs((2, 4, 64, 64)).cuda()
        feat = model(img)
        assert_tensor_with_shape(feat, torch.Size([2, 512, 2, 2]))

        # test resnet encoder with late downsample
        model = ResNetEnc('BasicBlock', [3, 4, 4, 2], 6, late_downsample=True)
        model.init_weights()
        model.train()
        model.cuda()
        # both image and trimap has 3 channels
        img = _demo_inputs((2, 6, 64, 64)).cuda()
        feat = model(img)
        assert_tensor_with_shape(feat, torch.Size([2, 512, 2, 2]))


def test_res_shortcut_encoder():
    """Test resnet encoder with shortcut."""
    with pytest.raises(NotImplementedError):
        ResShortcutEnc('UnknownBlock', [3, 4, 4, 2], 3)

    target_shape = [(2, 32, 64, 64), (2, 32, 32, 32), (2, 64, 16, 16),
                    (2, 128, 8, 8), (2, 256, 4, 4)]
    # target shape for model with late downsample
    target_late_ds_shape = [(2, 32, 64, 64), (2, 64, 32, 32), (2, 64, 16, 16),
                            (2, 128, 8, 8), (2, 256, 4, 4)]

    model = ResShortcutEnc(
        'BasicBlock', [3, 4, 4, 2], 4, with_spectral_norm=True)
    assert hasattr(model.conv1.conv, 'weight_orig')
    model.init_weights()
    model.train()
    # trimap has 1 channels
    img = _demo_inputs((2, 4, 64, 64))
    outputs = model(img)
    assert_tensor_with_shape(outputs['out'], (2, 512, 2, 2))
    assert_tensor_with_shape(outputs['feat1'], target_shape[0])
    assert_tensor_with_shape(outputs['feat2'], target_shape[1])
    assert_tensor_with_shape(outputs['feat3'], target_shape[2])
    assert_tensor_with_shape(outputs['feat4'], target_shape[3])
    assert_tensor_with_shape(outputs['feat5'], target_shape[4])

    model = ResShortcutEnc('BasicBlock', [3, 4, 4, 2], 6)
    model.init_weights()
    model.train()
    # both image and trimap has 3 channels
    img = _demo_inputs((2, 6, 64, 64))
    outputs = model(img)
    assert_tensor_with_shape(outputs['out'], (2, 512, 2, 2))
    assert_tensor_with_shape(outputs['feat1'], target_shape[0])
    assert_tensor_with_shape(outputs['feat2'], target_shape[1])
    assert_tensor_with_shape(outputs['feat3'], target_shape[2])
    assert_tensor_with_shape(outputs['feat4'], target_shape[3])
    assert_tensor_with_shape(outputs['feat5'], target_shape[4])

    # test resnet shortcut encoder with late downsample
    model = ResShortcutEnc('BasicBlock', [3, 4, 4, 2], 6, late_downsample=True)
    model.init_weights()
    model.train()
    # both image and trimap has 3 channels
    img = _demo_inputs((2, 6, 64, 64))
    outputs = model(img)
    assert_tensor_with_shape(outputs['out'], (2, 512, 2, 2))
    assert_tensor_with_shape(outputs['feat1'], target_late_ds_shape[0])
    assert_tensor_with_shape(outputs['feat2'], target_late_ds_shape[1])
    assert_tensor_with_shape(outputs['feat3'], target_late_ds_shape[2])
    assert_tensor_with_shape(outputs['feat4'], target_late_ds_shape[3])
    assert_tensor_with_shape(outputs['feat5'], target_late_ds_shape[4])

    if torch.cuda.is_available():
        # repeat above code again
        model = ResShortcutEnc(
            'BasicBlock', [3, 4, 4, 2], 4, with_spectral_norm=True)
        assert hasattr(model.conv1.conv, 'weight_orig')
        model.init_weights()
        model.train()
        model.cuda()
        # trimap has 1 channels
        img = _demo_inputs((2, 4, 64, 64)).cuda()
        outputs = model(img)
        assert_tensor_with_shape(outputs['out'], (2, 512, 2, 2))
        assert_tensor_with_shape(outputs['feat1'], target_shape[0])
        assert_tensor_with_shape(outputs['feat2'], target_shape[1])
        assert_tensor_with_shape(outputs['feat3'], target_shape[2])
        assert_tensor_with_shape(outputs['feat4'], target_shape[3])
        assert_tensor_with_shape(outputs['feat5'], target_shape[4])

        model = ResShortcutEnc('BasicBlock', [3, 4, 4, 2], 6)
        model.init_weights()
        model.train()
        model.cuda()
        # both image and trimap has 3 channels
        img = _demo_inputs((2, 6, 64, 64)).cuda()
        outputs = model(img)
        assert_tensor_with_shape(outputs['out'], (2, 512, 2, 2))
        assert_tensor_with_shape(outputs['feat1'], target_shape[0])
        assert_tensor_with_shape(outputs['feat2'], target_shape[1])
        assert_tensor_with_shape(outputs['feat3'], target_shape[2])
        assert_tensor_with_shape(outputs['feat4'], target_shape[3])
        assert_tensor_with_shape(outputs['feat5'], target_shape[4])

        # test resnet shortcut encoder with late downsample
        model = ResShortcutEnc(
            'BasicBlock', [3, 4, 4, 2], 6, late_downsample=True)
        model.init_weights()
        model.train()
        model.cuda()
        # both image and trimap has 3 channels
        img = _demo_inputs((2, 6, 64, 64)).cuda()
        outputs = model(img)
        assert_tensor_with_shape(outputs['out'], (2, 512, 2, 2))
        assert_tensor_with_shape(outputs['feat1'], target_late_ds_shape[0])
        assert_tensor_with_shape(outputs['feat2'], target_late_ds_shape[1])
        assert_tensor_with_shape(outputs['feat3'], target_late_ds_shape[2])
        assert_tensor_with_shape(outputs['feat4'], target_late_ds_shape[3])
        assert_tensor_with_shape(outputs['feat5'], target_late_ds_shape[4])


def test_res_gca_encoder():
    """Test resnet encoder with shortcut and guided contextual attention."""
    with pytest.raises(NotImplementedError):
        ResGCAEncoder('UnknownBlock', [3, 4, 4, 2], 3)

    target_shape = [(2, 32, 64, 64), (2, 32, 32, 32), (2, 64, 16, 16),
                    (2, 128, 8, 8), (2, 256, 4, 4)]
    # target shape for model with late downsample
    target_late_ds = [(2, 32, 64, 64), (2, 64, 32, 32), (2, 64, 16, 16),
                      (2, 128, 8, 8), (2, 256, 4, 4)]

    model = ResGCAEncoder('BasicBlock', [3, 4, 4, 2], 4)
    model.init_weights()
    model.train()
    # trimap has 1 channels
    img = _demo_inputs((2, 4, 64, 64))
    outputs = model(img)
    assert_tensor_with_shape(outputs['out'], (2, 512, 2, 2))
    assert_tensor_with_shape(outputs['img_feat'], (2, 128, 8, 8))
    assert_tensor_with_shape(outputs['unknown'], (2, 1, 8, 8))
    for i in range(5):
        assert_tensor_with_shape(outputs[f'feat{i+1}'], target_shape[i])

    model = ResGCAEncoder('BasicBlock', [3, 4, 4, 2], 6)
    model.init_weights()
    model.train()
    # both image and trimap has 3 channels
    img = _demo_inputs((2, 6, 64, 64))
    outputs = model(img)
    assert_tensor_with_shape(outputs['out'], (2, 512, 2, 2))
    assert_tensor_with_shape(outputs['img_feat'], (2, 128, 8, 8))
    assert_tensor_with_shape(outputs['unknown'], (2, 1, 8, 8))
    for i in range(5):
        assert_tensor_with_shape(outputs[f'feat{i+1}'], target_shape[i])

    # test resnet shortcut encoder with late downsample
    model = ResGCAEncoder('BasicBlock', [3, 4, 4, 2], 6, late_downsample=True)
    model.init_weights()
    model.train()
    # both image and trimap has 3 channels
    img = _demo_inputs((2, 6, 64, 64))
    outputs = model(img)
    assert_tensor_with_shape(outputs['out'], (2, 512, 2, 2))
    assert_tensor_with_shape(outputs['img_feat'], (2, 128, 8, 8))
    assert_tensor_with_shape(outputs['unknown'], (2, 1, 8, 8))
    for i in range(5):
        assert_tensor_with_shape(outputs[f'feat{i+1}'], target_late_ds[i])

    if torch.cuda.is_available():
        # repeat above code again
        model = ResGCAEncoder('BasicBlock', [3, 4, 4, 2], 4)
        model.init_weights()
        model.train()
        model.cuda()
        # trimap has 1 channels
        img = _demo_inputs((2, 4, 64, 64)).cuda()
        outputs = model(img)
        assert_tensor_with_shape(outputs['out'], (2, 512, 2, 2))
        assert_tensor_with_shape(outputs['img_feat'], (2, 128, 8, 8))
        assert_tensor_with_shape(outputs['unknown'], (2, 1, 8, 8))
        for i in range(5):
            assert_tensor_with_shape(outputs[f'feat{i+1}'], target_shape[i])

        model = ResGCAEncoder('BasicBlock', [3, 4, 4, 2], 6)
        model.init_weights()
        model.train()
        model.cuda()
        # both image and trimap has 3 channels
        img = _demo_inputs((2, 6, 64, 64)).cuda()
        outputs = model(img)
        assert_tensor_with_shape(outputs['out'], (2, 512, 2, 2))
        assert_tensor_with_shape(outputs['img_feat'], (2, 128, 8, 8))
        assert_tensor_with_shape(outputs['unknown'], (2, 1, 8, 8))
        for i in range(5):
            assert_tensor_with_shape(outputs[f'feat{i+1}'], target_shape[i])

        # test resnet shortcut encoder with late downsample
        model = ResGCAEncoder(
            'BasicBlock', [3, 4, 4, 2], 6, late_downsample=True)
        model.init_weights()
        model.train()
        model.cuda()
        # both image and trimap has 3 channels
        img = _demo_inputs((2, 6, 64, 64)).cuda()
        outputs = model(img)
        assert_tensor_with_shape(outputs['out'], (2, 512, 2, 2))
        assert_tensor_with_shape(outputs['img_feat'], (2, 128, 8, 8))
        assert_tensor_with_shape(outputs['unknown'], (2, 1, 8, 8))
        for i in range(5):
            assert_tensor_with_shape(outputs[f'feat{i+1}'], target_late_ds[i])


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


def _demo_inputs(input_shape=(2, 4, 64, 64)):
    """
    Create a superset of inputs needed to run encoder.

    Args:
        input_shape (tuple): input batch dimensions.
            Default: (1, 4, 64, 64).
    """
    img = np.random.random(input_shape).astype(np.float32)
    img = torch.from_numpy(img)

    return img
