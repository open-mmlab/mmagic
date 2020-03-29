import numpy as np
import torch
from mmedit.models.backbones import VGG16
from torch.nn.modules.batchnorm import _BatchNorm


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
    target_shape = [[2, 64, 32, 32], [2, 128, 16, 16], [2, 256, 8, 8],
                    [2, 512, 4, 4], [2, 512, 2, 2]]

    model = VGG16()
    model.init_weights()
    model.train()
    img = _demo_inputs()
    feat, mid_feat = model(img)
    assert_tensor_with_shape(feat, torch.Size([2, 512, 2, 2]))
    assert_mid_feat_shape(mid_feat, target_shape)

    model = VGG16(batch_norm=True)
    model.init_weights()
    model.train()
    img = _demo_inputs()
    feat, mid_feat = model(img)
    assert_tensor_with_shape(feat, torch.Size([2, 512, 2, 2]))
    assert_mid_feat_shape(mid_feat, target_shape)

    model = VGG16(aspp=True, atrous_rates=[6, 12, 18])
    model.init_weights()
    model.train()
    img = _demo_inputs()
    feat, mid_feat = model(img)
    assert_tensor_with_shape(feat, torch.Size([2, 256, 2, 2]))
    assert_mid_feat_shape(mid_feat, target_shape)
    assert check_norm_state(model.modules(), True)

    # test forward with gpu
    if torch.cuda.is_available():
        model = VGG16()
        model.init_weights()
        model.train()
        model.cuda()
        img = _demo_inputs().cuda()
        feat, mid_feat = model(img)
        assert_tensor_with_shape(feat, torch.Size([2, 512, 2, 2]))
        assert_mid_feat_shape(mid_feat, target_shape)

        model = VGG16(batch_norm=True)
        model.init_weights()
        model.train()
        model.cuda()
        img = _demo_inputs().cuda()
        feat, mid_feat = model(img)
        assert_tensor_with_shape(feat, torch.Size([2, 512, 2, 2]))
        assert_mid_feat_shape(mid_feat, target_shape)

        model = VGG16(aspp=True, atrous_rates=[6, 12, 18])
        model.init_weights()
        model.train()
        model.cuda()
        img = _demo_inputs().cuda()
        feat, mid_feat = model(img)
        assert_tensor_with_shape(feat, torch.Size([2, 256, 2, 2]))
        assert_mid_feat_shape(mid_feat, target_shape)
        assert check_norm_state(model.modules(), True)


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
