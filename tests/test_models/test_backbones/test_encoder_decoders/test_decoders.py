# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmedit.models.backbones import (VGG16, FBADecoder, IndexedUpsample,
                                     IndexNetDecoder, IndexNetEncoder,
                                     PlainDecoder, ResGCADecoder,
                                     ResGCAEncoder, ResNetDec, ResNetEnc,
                                     ResShortcutDec, ResShortcutEnc)


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


def test_plain_decoder():
    """Test PlainDecoder."""

    model = PlainDecoder(512)
    model.init_weights()
    model.train()
    # create max_pooling index for training
    encoder = VGG16(4)
    img = _demo_inputs()
    outputs = encoder(img)
    prediction = model(outputs)
    assert_tensor_with_shape(prediction, torch.Size([1, 1, 64, 64]))

    # test forward with gpu
    if torch.cuda.is_available():
        model = PlainDecoder(512)
        model.init_weights()
        model.train()
        model.cuda()
        encoder = VGG16(4)
        encoder.cuda()
        img = _demo_inputs().cuda()
        outputs = encoder(img)
        prediction = model(outputs)
        assert_tensor_with_shape(prediction, torch.Size([1, 1, 64, 64]))


def test_resnet_decoder():
    """Test resnet decoder."""
    with pytest.raises(NotImplementedError):
        ResNetDec('UnknowBlock', [2, 3, 3, 2], 512)

    model = ResNetDec('BasicBlockDec', [2, 3, 3, 2], 512, kernel_size=5)
    model.init_weights()
    model.train()
    encoder = ResNetEnc('BasicBlock', [2, 4, 4, 2], 6)
    img = _demo_inputs((1, 6, 64, 64))
    feat = encoder(img)
    prediction = model(feat)
    assert_tensor_with_shape(prediction, torch.Size([1, 1, 64, 64]))

    model = ResNetDec(
        'BasicBlockDec', [2, 3, 3, 2], 512, with_spectral_norm=True)
    assert hasattr(model.conv1.conv, 'weight_orig')
    model.init_weights()
    model.train()
    encoder = ResNetEnc('BasicBlock', [2, 4, 4, 2], 6)
    img = _demo_inputs((1, 6, 64, 64))
    feat = encoder(img)
    prediction = model(feat)
    assert_tensor_with_shape(prediction, torch.Size([1, 1, 64, 64]))

    # test forward with gpu
    if torch.cuda.is_available():
        model = ResNetDec('BasicBlockDec', [2, 3, 3, 2], 512, kernel_size=5)
        model.init_weights()
        model.train()
        model.cuda()
        encoder = ResNetEnc('BasicBlock', [2, 4, 4, 2], 6)
        encoder.cuda()
        img = _demo_inputs((1, 6, 64, 64)).cuda()
        feat = encoder(img)
        prediction = model(feat)
        assert_tensor_with_shape(prediction, torch.Size([1, 1, 64, 64]))

        model = ResNetDec(
            'BasicBlockDec', [2, 3, 3, 2], 512, with_spectral_norm=True)
        assert hasattr(model.conv1.conv, 'weight_orig')
        model.init_weights()
        model.train()
        model.cuda()
        encoder = ResNetEnc('BasicBlock', [2, 4, 4, 2], 6)
        encoder.cuda()
        img = _demo_inputs((1, 6, 64, 64)).cuda()
        feat = encoder(img)
        prediction = model(feat)
        assert_tensor_with_shape(prediction, torch.Size([1, 1, 64, 64]))


def test_res_shortcut_decoder():
    """Test resnet decoder with shortcut."""
    with pytest.raises(NotImplementedError):
        ResShortcutDec('UnknowBlock', [2, 3, 3, 2], 512)

    model = ResShortcutDec('BasicBlockDec', [2, 3, 3, 2], 512)
    model.init_weights()
    model.train()

    encoder = ResShortcutEnc('BasicBlock', [2, 4, 4, 2], 6)
    img = _demo_inputs((1, 6, 64, 64))
    outputs = encoder(img)
    prediction = model(outputs)
    assert_tensor_with_shape(prediction, torch.Size([1, 1, 64, 64]))

    # test forward with gpu
    if torch.cuda.is_available():
        model = ResShortcutDec('BasicBlockDec', [2, 3, 3, 2], 512)
        model.init_weights()
        model.train()
        model.cuda()
        encoder = ResShortcutEnc('BasicBlock', [2, 4, 4, 2], 6)
        encoder.cuda()
        img = _demo_inputs((1, 6, 64, 64)).cuda()
        outputs = encoder(img)
        prediction = model(outputs)
        assert_tensor_with_shape(prediction, torch.Size([1, 1, 64, 64]))


def test_res_gca_decoder():
    """Test resnet decoder with shortcut and guided contextual attention."""
    with pytest.raises(NotImplementedError):
        ResGCADecoder('UnknowBlock', [2, 3, 3, 2], 512)

    model = ResGCADecoder('BasicBlockDec', [2, 3, 3, 2], 512)
    model.init_weights()
    model.train()

    encoder = ResGCAEncoder('BasicBlock', [2, 4, 4, 2], 6)
    img = _demo_inputs((2, 6, 32, 32))
    outputs = encoder(img)
    prediction = model(outputs)
    assert_tensor_with_shape(prediction, torch.Size([2, 1, 32, 32]))

    # test forward with gpu
    if torch.cuda.is_available():
        model = ResGCADecoder('BasicBlockDec', [2, 3, 3, 2], 512)
        model.init_weights()
        model.train()
        model.cuda()
        encoder = ResGCAEncoder('BasicBlock', [2, 4, 4, 2], 6)
        encoder.cuda()
        img = _demo_inputs((2, 6, 32, 32)).cuda()
        outputs = encoder(img)
        prediction = model(outputs)
        assert_tensor_with_shape(prediction, torch.Size([2, 1, 32, 32]))


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


def test_indexnet_decoder():
    """Test Indexnet decoder."""
    # test indexnet decoder with default indexnet setting
    with pytest.raises(AssertionError):
        # shortcut must have four dimensions
        indexnet_decoder = IndexNetDecoder(
            160, kernel_size=5, separable_conv=False)
        x = torch.rand(2, 256, 4, 4)
        shortcut = torch.rand(2, 128, 8, 8, 8)
        dec_idx_feat = torch.rand(2, 128, 8, 8, 8)
        outputs_enc = dict(
            out=x, shortcuts=[shortcut], dec_idx_feat_list=[dec_idx_feat])
        indexnet_decoder(outputs_enc)

    indexnet_decoder = IndexNetDecoder(
        160, kernel_size=5, separable_conv=False)
    indexnet_decoder.init_weights()
    indexnet_encoder = IndexNetEncoder(4)
    x = torch.rand(2, 4, 32, 32)
    outputs_enc = indexnet_encoder(x)
    out = indexnet_decoder(outputs_enc)
    assert out.shape == (2, 1, 32, 32)

    # test indexnet decoder with other setting
    indexnet_decoder = IndexNetDecoder(160, kernel_size=3, separable_conv=True)
    indexnet_decoder.init_weights()
    out = indexnet_decoder(outputs_enc)
    assert out.shape == (2, 1, 32, 32)


def test_fba_decoder():

    with pytest.raises(AssertionError):
        # pool_scales must be list|tuple
        FBADecoder(pool_scales=1, in_channels=32, channels=16)
    inputs = dict()
    conv_out_1 = _demo_inputs((1, 11, 320, 320))
    conv_out_2 = _demo_inputs((1, 64, 160, 160))
    conv_out_3 = _demo_inputs((1, 256, 80, 80))
    conv_out_4 = _demo_inputs((1, 512, 40, 40))
    conv_out_5 = _demo_inputs((1, 1024, 40, 40))
    conv_out_6 = _demo_inputs((1, 2048, 40, 40))
    inputs['conv_out'] = [
        conv_out_1, conv_out_2, conv_out_3, conv_out_4, conv_out_5, conv_out_6
    ]
    inputs['merged'] = _demo_inputs((1, 3, 320, 320))
    inputs['two_channel_trimap'] = _demo_inputs((1, 2, 320, 320))
    model = FBADecoder(
        pool_scales=(1, 2, 3, 6),
        in_channels=2048,
        channels=256,
        norm_cfg=dict(type='GN', num_groups=32))

    alpha, F, B = model(inputs)
    assert_tensor_with_shape(alpha, torch.Size([1, 1, 320, 320]))
    assert_tensor_with_shape(F, torch.Size([1, 3, 320, 320]))
    assert_tensor_with_shape(B, torch.Size([1, 3, 320, 320]))
