# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

from mmedit.models.backbones import SimpleEncoderDecoder


def assert_dict_keys_equal(dictionary, target_keys):
    """Check if the keys of the dictionary is equal to the target key set."""
    assert isinstance(dictionary, dict)
    assert set(dictionary.keys()) == set(target_keys)


def assert_tensor_with_shape(tensor, shape):
    """"Check if the shape of the tensor is equal to the target shape."""
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == shape


def test_encoder_decoder():
    """Test SimpleEncoderDecoder."""
    # check DIM with only alpha loss
    encoder = dict(type='VGG16', in_channels=4)
    decoder = dict(type='PlainDecoder')

    model = SimpleEncoderDecoder(encoder, decoder)
    model.init_weights()
    model.train()
    fg, bg, merged, alpha, trimap = _demo_inputs_pair()
    prediction = model(torch.cat([merged, trimap], 1))
    assert_tensor_with_shape(prediction, torch.Size([1, 1, 64, 64]))

    # check DIM with only composition loss
    encoder = dict(type='VGG16', in_channels=4)
    decoder = dict(type='PlainDecoder')

    model = SimpleEncoderDecoder(encoder, decoder)
    model.init_weights()
    model.train()
    fg, bg, merged, alpha, trimap = _demo_inputs_pair()
    prediction = model(torch.cat([merged, trimap], 1))
    assert_tensor_with_shape(prediction, torch.Size([1, 1, 64, 64]))

    # check DIM with both alpha and composition loss
    encoder = dict(type='VGG16', in_channels=4)
    decoder = dict(type='PlainDecoder')
    model = SimpleEncoderDecoder(encoder, decoder)
    model.init_weights()
    model.train()
    fg, bg, merged, alpha, trimap = _demo_inputs_pair()
    prediction = model(torch.cat([merged, trimap], 1))
    assert_tensor_with_shape(prediction, torch.Size([1, 1, 64, 64]))

    # test forward with gpu
    if torch.cuda.is_available():
        encoder = dict(type='VGG16', in_channels=4)
        decoder = dict(type='PlainDecoder')

        model = SimpleEncoderDecoder(encoder, decoder)
        model.init_weights()
        model.train()
        fg, bg, merged, alpha, trimap = _demo_inputs_pair(cuda=True)
        model.cuda()
        prediction = model(torch.cat([merged, trimap], 1))
        assert_tensor_with_shape(prediction, torch.Size([1, 1, 64, 64]))


def _demo_inputs_pair(img_shape=(64, 64), batch_size=1, cuda=False):
    """Create a superset of inputs needed to run backbone.

    Args:
        img_shape (tuple): shape of the input image.
        batch_size (int): batch size of the input batch.
        cuda (bool): whether transfer input into gpu.
    """
    color_shape = (batch_size, 3, img_shape[0], img_shape[1])
    gray_shape = (batch_size, 1, img_shape[0], img_shape[1])
    fg = torch.from_numpy(np.random.random(color_shape).astype(np.float32))
    bg = torch.from_numpy(np.random.random(color_shape).astype(np.float32))
    merged = torch.from_numpy(np.random.random(color_shape).astype(np.float32))
    alpha = torch.from_numpy(np.random.random(gray_shape).astype(np.float32))
    trimap = torch.from_numpy(np.random.random(gray_shape).astype(np.float32))
    if cuda:
        fg = fg.cuda()
        bg = bg.cuda()
        merged = merged.cuda()
        alpha = alpha.cuda()
        trimap = trimap.cuda()
    return fg, bg, merged, alpha, trimap
