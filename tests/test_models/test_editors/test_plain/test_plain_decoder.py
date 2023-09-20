# Copyright (c) OpenMMLab. All rights reserved.
import platform

import numpy as np
import pytest
import torch

from mmagic.models.archs import VGG16
from mmagic.models.editors import PlainDecoder


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


@pytest.mark.skipif(
    'win' in platform.system().lower() and 'cu' in torch.__version__,
    reason='skip on windows-cuda due to limited RAM.')
def test_plain_decoder():
    """Test PlainDecoder."""

    with torch.no_grad():
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


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
