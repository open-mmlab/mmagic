# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

from mmagic.models.editors.cyclegan.cyclegan_modules import (
    GANImageBuffer, ResidualBlockWithDropout)


def test_residual_block_with_dropout():
    block = ResidualBlockWithDropout(16, 'zeros')
    input = torch.rand((2, 16, 128, 128))
    output = block(input)
    assert output.detach().numpy().shape == (2, 16, 128, 128)

    block = ResidualBlockWithDropout(16, 'zeros', use_dropout=False)
    assert len(block.block) == 2


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

    # set buffer ratio as 1 and 0 to cover more lines
    buffer = GANImageBuffer(buffer_size=1, buffer_ratio=1)
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

    buffer = GANImageBuffer(buffer_size=1, buffer_ratio=0)
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


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
