import numpy as np
import pytest
import torch
from mmedit.models.backbones import (VGG16, BGMattingDecoder, IndexedUpsample,
                                     IndexNetDecoder, IndexNetEncoder,
                                     PlainDecoder, ResNetDec, ResNetEnc,
                                     ResShortcutDec, ResShortcutEnc)


def assert_tensor_with_shape(tensor, shape):
    """"Check if the shape of the tensor is equal to the target shape."""
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == shape


def test_plain_decoder():
    """Test PlainDecoder."""

    model = PlainDecoder(512)
    model.init_weights()
    model.train()
    # create max_pooling index for training
    encoder = VGG16()
    img = _demo_inputs()
    feat, mid_feat = encoder(img)
    prediction = model(feat, mid_feat)
    assert_tensor_with_shape(prediction, torch.Size([1, 1, 64, 64]))

    # test forward with gpu
    if torch.cuda.is_available():
        model = PlainDecoder(512)
        model.init_weights()
        model.train()
        model.cuda()
        encoder = VGG16()
        encoder.cuda()
        img = _demo_inputs().cuda()
        feat, mid_feat = encoder(img)
        prediction = model(feat, mid_feat)
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
    indexnet_encoder = IndexNetEncoder()
    x = torch.rand(2, 4, 32, 32)
    outputs_enc = indexnet_encoder(x)
    out = indexnet_decoder(outputs_enc)
    assert out.shape == (2, 1, 32, 32)

    # test indexnet decoder with other setting
    indexnet_decoder = IndexNetDecoder(160, kernel_size=3, separable_conv=True)
    indexnet_decoder.init_weights()
    out = indexnet_decoder(outputs_enc)
    assert out.shape == (2, 1, 32, 32)


def test_bg_matting_decoder():
    """Test BGMattingDecoder."""
    out = _demo_inputs((2, 448, 8, 8))
    img_feat1 = _demo_inputs((2, 128, 16, 16))
    inputs = {'out': out, 'img_feat1': img_feat1}
    bg_matting_decoder = BGMattingDecoder(448)
    bg_matting_decoder.init_weights()
    outputs = bg_matting_decoder(inputs)
    assert_tensor_with_shape(outputs['alpha_out'], (2, 1, 32, 32))
    assert_tensor_with_shape(outputs['fg_out'], (2, 3, 32, 32))

    bg_matting_decoder = BGMattingDecoder(448, use_dropout=True)
    bg_matting_decoder.init_weights()
    outputs = bg_matting_decoder(inputs)
    assert_tensor_with_shape(outputs['alpha_out'], (2, 1, 32, 32))
    assert_tensor_with_shape(outputs['fg_out'], (2, 3, 32, 32))


def _demo_inputs(input_shape=(1, 4, 64, 64)):
    """
    Create a superset of inputs needed to run encoder.

    Args:
        input_shape (tuple): input batch dimensions.
            Default: (1, 4, 64, 64).
    """
    img = np.random.random(input_shape).astype(np.float32)
    img = torch.from_numpy(img)

    return img
