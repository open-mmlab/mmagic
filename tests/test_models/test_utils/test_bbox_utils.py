# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmagic.models.editors import PlainRefiner
from mmagic.models.utils import extract_around_bbox, extract_bbox_patch


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


def assert_dict_keys_equal(dictionary, target_keys):
    """Check if the keys of the dictionary is equal to the target key set."""
    assert isinstance(dictionary, dict)
    assert set(dictionary.keys()) == set(target_keys)


def assert_tensor_with_shape(tensor, shape):
    """"Check if the shape of the tensor is equal to the target shape."""
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == shape


def test_plain_refiner():
    """Test PlainRefiner."""
    model = PlainRefiner()
    model.init_weights()
    model.train()
    merged, alpha, trimap, raw_alpha = _demo_inputs_pair()
    prediction = model(torch.cat([merged, raw_alpha.sigmoid()], 1), raw_alpha)
    assert_tensor_with_shape(prediction, torch.Size([1, 1, 64, 64]))

    # test forward with gpu
    if torch.cuda.is_available():
        model = PlainRefiner()
        model.init_weights()
        model.train()
        model.cuda()
        merged, alpha, trimap, raw_alpha = _demo_inputs_pair(cuda=True)
        prediction = model(
            torch.cat([merged, raw_alpha.sigmoid()], 1), raw_alpha)
        assert_tensor_with_shape(prediction, torch.Size([1, 1, 64, 64]))


def _demo_inputs_pair(img_shape=(64, 64), batch_size=1, cuda=False):
    """Create a superset of inputs needed to run refiner.

    Args:
        img_shape (tuple): shape of the input image.
        batch_size (int): batch size of the input batch.
        cuda (bool): whether transfer input into gpu.
    """
    color_shape = (batch_size, 3, img_shape[0], img_shape[1])
    gray_shape = (batch_size, 1, img_shape[0], img_shape[1])
    merged = torch.from_numpy(np.random.random(color_shape).astype(np.float32))
    alpha = torch.from_numpy(np.random.random(gray_shape).astype(np.float32))
    trimap = torch.from_numpy(np.random.random(gray_shape).astype(np.float32))
    raw_alpha = torch.from_numpy(
        np.random.random(gray_shape).astype(np.float32))
    if cuda:
        merged = merged.cuda()
        alpha = alpha.cuda()
        trimap = trimap.cuda()
        raw_alpha = raw_alpha.cuda()
    return merged, alpha, trimap, raw_alpha


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
