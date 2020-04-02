import numpy as np
import pytest
import torch
from mmedit.models.backbones import VGG16, PlainDecoder, ResNetDec, ResNetEnc


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

    model = ResNetDec('BasicBlockDec', [2, 3, 3, 2], 512)
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

        model = ResNetDec('BasicBlockDec', [2, 3, 3, 2], 512)
        model.init_weights()
        model.train()
        model.cuda()
        encoder = ResNetEnc('BasicBlock', [2, 4, 4, 2], 6)
        encoder.cuda()
        img = _demo_inputs((1, 6, 64, 64)).cuda()
        feat = encoder(img)
        prediction = model(feat)
        assert_tensor_with_shape(prediction, torch.Size([1, 1, 64, 64]))


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
