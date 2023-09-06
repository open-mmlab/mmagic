# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmagic.models.editors.gca import ResNetEnc, ResShortcutEnc


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


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
