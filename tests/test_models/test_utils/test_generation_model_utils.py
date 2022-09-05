# Copyright (c) OpenMMLab. All rights reserved.
import copy

import numpy as np
import pytest
import torch
import torch.nn as nn

from mmedit.models.utils import (GANImageBuffer, ResidualBlockWithDropout,
                                 UnetSkipConnectionBlock,
                                 generation_init_weights)


def test_gan_image_buffer():
    # test buffer size = 0
    buffer = GANImageBuffer(buffer_size=0)
    img_np = np.random.randn(1, 3, 256, 256)
    img_tensor = torch.from_numpy(img_np)
    img_tensor_return = buffer.query(img_tensor)
    assert torch.equal(img_tensor_return, img_tensor)

    # test buffer size > 0
    buffer = GANImageBuffer(buffer_size=1)
    img_np = np.random.randn(2, 3, 256, 256)
    img_tensor = torch.from_numpy(img_np)
    img_tensor_0 = torch.unsqueeze(img_tensor[0], 0)
    img_tensor_1 = torch.unsqueeze(img_tensor[1], 0)
    img_tensor_00 = torch.cat([img_tensor_0, img_tensor_0], 0)
    img_tensor_return = buffer.query(img_tensor)
    assert (torch.equal(img_tensor_return, img_tensor)
            and torch.equal(buffer.image_buffer[0], img_tensor_0)) or \
           (torch.equal(img_tensor_return, img_tensor_00)
            and torch.equal(buffer.image_buffer[0], img_tensor_1))

    # test buffer size > 0, specify buffer chance
    buffer = GANImageBuffer(buffer_size=1, buffer_ratio=0.3)
    img_np = np.random.randn(2, 3, 256, 256)
    img_tensor = torch.from_numpy(img_np)
    img_tensor_0 = torch.unsqueeze(img_tensor[0], 0)
    img_tensor_1 = torch.unsqueeze(img_tensor[1], 0)
    img_tensor_00 = torch.cat([img_tensor_0, img_tensor_0], 0)
    img_tensor_return = buffer.query(img_tensor)
    assert (torch.equal(img_tensor_return, img_tensor)
            and torch.equal(buffer.image_buffer[0], img_tensor_0)) or \
           (torch.equal(img_tensor_return, img_tensor_00)
            and torch.equal(buffer.image_buffer[0], img_tensor_1))


def test_generation_init_weights():
    # Conv
    module = nn.Conv2d(3, 3, 1)
    module_tmp = copy.deepcopy(module)
    generation_init_weights(module, init_type='normal', init_gain=0.02)
    generation_init_weights(module, init_type='xavier', init_gain=0.02)
    generation_init_weights(module, init_type='kaiming')
    generation_init_weights(module, init_type='orthogonal', init_gain=0.02)
    with pytest.raises(NotImplementedError):
        generation_init_weights(module, init_type='abc')
    assert not torch.equal(module.weight.data, module_tmp.weight.data)

    # Linear
    module = nn.Linear(3, 1)
    module_tmp = copy.deepcopy(module)
    generation_init_weights(module, init_type='normal', init_gain=0.02)
    generation_init_weights(module, init_type='xavier', init_gain=0.02)
    generation_init_weights(module, init_type='kaiming')
    generation_init_weights(module, init_type='orthogonal', init_gain=0.02)
    with pytest.raises(NotImplementedError):
        generation_init_weights(module, init_type='abc')
    assert not torch.equal(module.weight.data, module_tmp.weight.data)

    # BatchNorm2d
    module = nn.BatchNorm2d(3)
    module_tmp = copy.deepcopy(module)
    generation_init_weights(module, init_type='normal', init_gain=0.02)
    assert not torch.equal(module.weight.data, module_tmp.weight.data)


def test_unet_skip_connection_block():
    block = UnetSkipConnectionBlock(16, 16, is_innermost=True)
    input = torch.rand((2, 16, 128, 128))
    output = block(input)
    assert output.detach().numpy().shape == (2, 32, 128, 128)


def test_residual_block_with_dropout():
    block = ResidualBlockWithDropout(16, 'zeros')
    input = torch.rand((2, 16, 128, 128))
    output = block(input)
    assert output.detach().numpy().shape == (2, 16, 128, 128)
