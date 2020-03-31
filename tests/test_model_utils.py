import numpy as np
import pytest
import torch
from mmedit.models.common import (extract_around_bbox, extract_bbox_patch,
                                  set_requires_grad)


def test_set_requires_grad():
    model = torch.nn.Conv2d(1, 3, 1, 1)
    set_requires_grad(model, False)
    for param in model.parameters():
        assert not param.requires_grad


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
