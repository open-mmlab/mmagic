# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmedit.datasets.pipelines.utils import (adjust_gamma, dtype_range,
                                             make_coord)


def test_adjust_gamma():
    """Test Gamma Correction.

    Adpted from
    # https://github.com/scikit-image/scikit-image/blob/7e4840bd9439d1dfb6beaf549998452c99f97fdd/skimage/exposure/tests/test_exposure.py#L534  # noqa
    """
    # Check that the shape is maintained.
    img = np.ones([1, 1])
    result = adjust_gamma(img, 1.5)
    assert img.shape == result.shape

    # Same image should be returned for gamma equal to one.
    image = np.random.uniform(0, 255, (8, 8))
    result = adjust_gamma(image, 1)
    np.testing.assert_array_equal(result, image)

    # White image should be returned for gamma equal to zero.
    image = np.random.uniform(0, 255, (8, 8))
    result = adjust_gamma(image, 0)
    dtype = image.dtype.type
    np.testing.assert_array_equal(result, dtype_range[dtype][1])

    # Verifying the output with expected results for gamma
    # correction with gamma equal to half.
    image = np.arange(0, 255, 4, np.uint8).reshape((8, 8))
    expected = np.array([[0, 31, 45, 55, 63, 71, 78, 84],
                         [90, 95, 100, 105, 110, 115, 119, 123],
                         [127, 131, 135, 139, 142, 146, 149, 153],
                         [156, 159, 162, 165, 168, 171, 174, 177],
                         [180, 183, 186, 188, 191, 194, 196, 199],
                         [201, 204, 206, 209, 211, 214, 216, 218],
                         [221, 223, 225, 228, 230, 232, 234, 236],
                         [238, 241, 243, 245, 247, 249, 251, 253]],
                        dtype=np.uint8)

    result = adjust_gamma(image, 0.5)
    np.testing.assert_array_equal(result, expected)

    # Verifying the output with expected results for gamma
    # correction with gamma equal to two.
    image = np.arange(0, 255, 4, np.uint8).reshape((8, 8))
    expected = np.array([[0, 0, 0, 0, 1, 1, 2, 3], [4, 5, 6, 7, 9, 10, 12, 14],
                         [16, 18, 20, 22, 25, 27, 30, 33],
                         [36, 39, 42, 45, 49, 52, 56, 60],
                         [64, 68, 72, 76, 81, 85, 90, 95],
                         [100, 105, 110, 116, 121, 127, 132, 138],
                         [144, 150, 156, 163, 169, 176, 182, 189],
                         [196, 203, 211, 218, 225, 233, 241, 249]],
                        dtype=np.uint8)

    result = adjust_gamma(image, 2)
    np.testing.assert_array_equal(result, expected)

    # Test invalid image input
    image = np.arange(0, 255, 4, np.uint8).reshape((8, 8))
    with pytest.raises(ValueError):
        adjust_gamma(image, -1)


def test_make_coord():
    h, w = 20, 30

    coord = make_coord((h, w), ranges=((10, 20), (-5, 5)))
    assert type(coord) == torch.Tensor
    assert coord.shape == (h * w, 2)

    coord = make_coord((h, w), flatten=False)
    assert type(coord) == torch.Tensor
    assert coord.shape == (h, w, 2)


test_make_coord()
