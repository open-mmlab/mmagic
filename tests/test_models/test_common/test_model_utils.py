# Copyright (c) OpenMMLab. All rights reserved.
import copy

import numpy as np
import pytest
import torch
import torch.nn as nn

from mmedit.models.common import (GANImageBuffer, extract_around_bbox,
                                  extract_bbox_patch, generation_init_weights,
                                  set_requires_grad)


def test_set_requires_grad():
    model = torch.nn.Conv2d(1, 3, 1, 1)
    set_requires_grad(model, False)
    for param in model.parameters():
        assert not param.requires_grad


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


def test_extract_bbox_patch():
    img_np = np.random.randn(100, 100, 3)
    bbox = np.asarray([10, 10, 10, 10])
    img_patch = extract_bbox_patch(bbox, img_np, channel_first=False)
    assert np.array_equal(img_patch, img_np[10:20, 10:20, ...])

    img_np = np.random.randn(1, 3, 100, 100)
    bbox = np.asarray([[10, 10, 10, 10]])
    img_patch = extract_bbox_patch(bbox, img_np)
    assert np.array_equal(img_patch, img_np[..., 10:20, 10:20])

    img_tensor = torch.from_numpy(img_np)
    bbox = np.asarray([[10, 10, 10, 10]])
    img_patch = extract_bbox_patch(bbox, img_tensor)
    assert np.array_equal(img_patch.numpy(), img_np[..., 10:20, 10:20])

    with pytest.raises(AssertionError):
        img_np = np.random.randn(100, 100)
        bbox = np.asarray([[10, 10, 10, 10]])
        img_patch = extract_bbox_patch(bbox, img_np)

    with pytest.raises(AssertionError):
        img_np = np.random.randn(2, 3, 100, 100)
        bbox = np.asarray([[10, 10, 10, 10]])
        img_patch = extract_bbox_patch(bbox, img_np)

    with pytest.raises(AssertionError):
        img_np = np.random.randn(3, 100, 100)
        bbox = np.asarray([[10, 10, 10, 10]])
        img_patch = extract_bbox_patch(bbox, img_np)


def test_extract_around_bbox():
    with pytest.raises(AssertionError):
        img_np = np.random.randn(100, 100, 3)
        bbox = np.asarray([10, 10, 10, 10])
        extract_around_bbox(img_np, bbox, (4, 4))

    with pytest.raises(TypeError):
        bbox = dict(test='fail')
        img_np = np.random.randn(100, 100, 3)
        extract_around_bbox(img_np, bbox, (15, 15))

    img_np = np.random.randn(100, 100, 3)
    bbox = np.asarray([10, 10, 10, 10])
    img_new, bbox_new = extract_around_bbox(
        img_np, bbox, (14, 14), channel_first=False)
    assert np.array_equal(img_np[8:22, 8:22, ...], img_new)
    assert np.array_equal(bbox_new, np.asarray([8, 8, 14, 14]))

    img_np = np.random.randn(1, 3, 100, 100)
    bbox = np.asarray([[10, 10, 10, 10]])
    img_tensor = torch.from_numpy(img_np)
    bbox_tensor = torch.from_numpy(bbox)
    img_new, bbox_new = extract_around_bbox(
        img_tensor, bbox_tensor, target_size=[14, 14])
    assert np.array_equal(img_np[..., 8:22, 8:22], img_new.numpy())
    assert np.array_equal(bbox_new.numpy(), np.asarray([[8, 8, 14, 14]]))
