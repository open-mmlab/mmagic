# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmagic.models.editors.gca import GCAModule, ResGCAEncoder


def _demo_inputs(input_shape=(1, 4, 64, 64)):
    """Create a superset of inputs needed to run encoder.

    Args:
        input_shape (tuple): input batch dimensions.
            Default: (1, 4, 64, 64).
    """
    img = np.random.random(input_shape).astype(np.float32)
    img = torch.from_numpy(img)

    return img


def assert_tensor_with_shape(tensor, shape):
    """"Check if the shape of the tensor is equal to the target shape."""
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == shape


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


def test_gca_module():
    img_feat = torch.rand(1, 128, 64, 64)
    alpha_feat = torch.rand(1, 128, 64, 64)
    unknown = None
    gca = GCAModule(128, 128, rate=1)
    output = gca(img_feat, alpha_feat, unknown)
    assert output.shape == (1, 128, 64, 64)

    img_feat = torch.rand(1, 128, 64, 64)
    alpha_feat = torch.rand(1, 128, 64, 64)
    unknown = torch.rand(1, 1, 64, 64)
    gca = GCAModule(128, 128, rate=2)
    output = gca(img_feat, alpha_feat, unknown)
    assert output.shape == (1, 128, 64, 64)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
