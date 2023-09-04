# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmagic.models.editors.gca import (ResGCADecoder, ResGCAEncoder, ResNetDec,
                                       ResNetEnc, ResShortcutDec,
                                       ResShortcutEnc)


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


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
