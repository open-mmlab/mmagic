# Copyright (c) OpenMMLab. All rights reserved.

import cv2
import numpy as np
import pytest

from mmagic.datasets.transforms import (FormatTrimap, GenerateTrimap,
                                        GenerateTrimapWithDistTransform,
                                        TransformTrimap)


def assert_keys_contain(result_keys, target_keys):
    """Check if all elements in target_keys is in result_keys."""
    assert set(target_keys).issubset(set(result_keys))


def generate_ref_trimap(alpha, kernel_size, iterations, random):
    """Check if a trimap's value is correct."""
    if isinstance(kernel_size, int):
        kernel_size = kernel_size, kernel_size + 1
    if isinstance(iterations, int):
        iterations = iterations, iterations + 1

    if random:
        min_kernel, max_kernel = kernel_size
        kernel_num = max_kernel - min_kernel
        erode_ksize = min_kernel + np.random.randint(kernel_num)
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                 (erode_ksize, erode_ksize))
        dilate_ksize = min_kernel + np.random.randint(kernel_num)
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                  (dilate_ksize, dilate_ksize))

        min_iteration, max_iteration = iterations
        erode_iter = np.random.randint(min_iteration, max_iteration)
        dilate_iter = np.random.randint(min_iteration, max_iteration)
    else:
        erode_ksize, dilate_ksize = kernel_size
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                 (erode_ksize, erode_ksize))
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                  (dilate_ksize, dilate_ksize))
        erode_iter, dilate_iter = iterations

    h, w = alpha.shape

    # erode
    erode_kh = erode_kw = erode_ksize
    eroded = np.zeros_like(alpha)
    src = alpha
    pad = ((erode_kh // 2, (erode_kh - 1) // 2), (erode_kw // 2,
                                                  (erode_kw - 1) // 2))
    for _ in range(erode_iter):
        src = np.pad(src, pad, 'constant', constant_values=np.max(src))
        for i in range(h):
            for j in range(w):
                target = src[i:i + erode_kh, j:j + erode_kw]
                eroded[i, j] = np.min(
                    (target * erode_kernel)[erode_kernel == 1])
        src = eroded

    # dilate
    dilate_kh = dilate_kw = dilate_ksize
    dilated = np.zeros_like(alpha)
    src = alpha
    pad = ((dilate_kh // 2, (dilate_kh - 1) // 2), (dilate_kw // 2,
                                                    (dilate_kw - 1) // 2))
    for _ in range(dilate_iter):
        src = np.pad(src, pad, constant_values=np.min(src))
        for i in range(h):
            for j in range(w):
                target = src[i:i + dilate_kh, j:j + dilate_kw]
                dilated[i, j] = np.max(
                    (target * dilate_kernel)[dilate_kernel == 1])
        src = dilated

    ref_trimap = np.zeros_like(alpha)
    ref_trimap.fill(128)
    ref_trimap[eroded >= 255] = 255
    ref_trimap[dilated <= 0] = 0
    return ref_trimap


def test_format_trimap():
    ori_trimap = np.random.randint(3, size=(64, 64))
    ori_trimap[ori_trimap == 1] = 128
    ori_trimap[ori_trimap == 2] = 255

    results = dict(trimap=ori_trimap.copy())
    format_trimap = FormatTrimap(to_onehot=False)
    results = format_trimap(results)
    result_trimap = results['trimap']
    assert repr(format_trimap) == format_trimap.__class__.__name__ + (
        '(to_onehot=False)')
    assert result_trimap.shape == (64, 64)
    assert ((result_trimap == 0) == (ori_trimap == 0)).all()
    assert ((result_trimap == 1) == (ori_trimap == 128)).all()
    assert ((result_trimap == 2) == (ori_trimap == 255)).all()
    assert results['format_trimap_to_onehot'] is False

    results = dict(trimap=ori_trimap.copy())
    format_trimap = FormatTrimap(to_onehot=True)
    results = format_trimap(results)
    result_trimap = results['trimap']
    assert repr(format_trimap) == format_trimap.__class__.__name__ + (
        '(to_onehot=True)')
    assert result_trimap.shape == (64, 64, 3)
    assert ((result_trimap[..., 0] == 1) == (ori_trimap == 0)).all()
    assert ((result_trimap[..., 1] == 1) == (ori_trimap == 128)).all()
    assert ((result_trimap[..., 2] == 1) == (ori_trimap == 255)).all()
    assert results['format_trimap_to_onehot'] is True


def test_generate_trimap():

    with pytest.raises(ValueError):
        # kernel_size must be an int or a tuple of 2 int
        GenerateTrimap(1.5)

    with pytest.raises(ValueError):
        # kernel_size must be an int or a tuple of 2 int
        GenerateTrimap((3, 3, 3))

    with pytest.raises(ValueError):
        # iterations must be an int or a tuple of 2 int
        GenerateTrimap(3, iterations=1.5)

    with pytest.raises(ValueError):
        # iterations must be an int or a tuple of 2 int
        GenerateTrimap(3, iterations=(3, 3, 3))

    target_keys = ['alpha', 'trimap']

    # check random mode
    kernel_size = (3, 5)
    iterations = (3, 5)
    random = True
    alpha = np.random.randn(32, 32)
    results = dict(alpha=alpha)
    generate_trimap = GenerateTrimap(kernel_size, iterations, random)
    np.random.seed(123)
    generate_trimap_results = generate_trimap(results)
    trimap = generate_trimap_results['trimap']

    assert_keys_contain(generate_trimap_results.keys(), target_keys)
    assert trimap.shape == alpha.shape
    np.random.seed(123)
    ref_trimap = generate_ref_trimap(alpha, kernel_size, iterations, random)
    assert (trimap == ref_trimap).all()

    # check non-random mode
    kernel_size = (3, 5)
    iterations = (5, 3)
    random = False
    generate_trimap = GenerateTrimap(kernel_size, iterations, random)
    generate_trimap_results = generate_trimap(results)
    trimap = generate_trimap_results['trimap']

    assert_keys_contain(generate_trimap_results.keys(), target_keys)
    assert trimap.shape == alpha.shape
    ref_trimap = generate_ref_trimap(alpha, kernel_size, iterations, random)
    assert (trimap == ref_trimap).all()

    # check repr string
    kernel_size = 1
    iterations = 1
    generate_trimap = GenerateTrimap(kernel_size, iterations)
    kernels = [
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (kernel_size, kernel_size))
    ]
    assert repr(generate_trimap) == (
        generate_trimap.__class__.__name__ +
        f'(kernels={kernels}, iterations={(iterations, iterations + 1)}, '
        f'random=True)')


def test_generate_trimap_with_dist_transform():
    with pytest.raises(ValueError):
        # dist_thr must be an float that is greater than 1
        GenerateTrimapWithDistTransform(dist_thr=-1)

    target_keys = ['alpha', 'trimap']

    alpha = np.random.randint(0, 256, (32, 32))
    alpha[:8, :8] = 0
    alpha[-8:, -8:] = 255
    results = dict(alpha=alpha)
    generate_trimap = GenerateTrimapWithDistTransform(dist_thr=3, random=False)
    generate_trimap_results = generate_trimap(results)
    trimap = generate_trimap_results['trimap']
    assert_keys_contain(generate_trimap_results.keys(), target_keys)
    assert trimap.shape == alpha.shape

    alpha = np.random.randint(0, 256, (32, 32))
    results = dict(alpha=alpha)
    generate_trimap = GenerateTrimapWithDistTransform(dist_thr=3, random=True)
    generate_trimap_results = generate_trimap(results)
    trimap = generate_trimap_results['trimap']
    assert_keys_contain(generate_trimap_results.keys(), target_keys)
    assert trimap.shape == alpha.shape

    assert repr(generate_trimap) == (
        generate_trimap.__class__.__name__ + '(dist_thr=3, random=True)')


def test_transform_trimap():
    results = dict()
    transform = TransformTrimap()
    target_keys = ['trimap', 'transformed_trimap']

    with pytest.raises(KeyError):
        results_transformed = transform(results)

    with pytest.raises(AssertionError):
        dummy_trimap = np.zeros((100, 100, 1), dtype=np.uint8)
        results['trimap'] = dummy_trimap
        results_transformed = transform(results)

    results = dict()
    # generate dummy trimap with shape (100,100)
    dummy_trimap = np.zeros((100, 100), dtype=np.uint8)
    dummy_trimap[:50, :50] = 255
    results['trimap'] = dummy_trimap
    results_transformed = transform(results)
    assert_keys_contain(results_transformed.keys(), target_keys)
    assert results_transformed['trimap'].shape == dummy_trimap.shape
    assert results_transformed[
        'transformed_trimap'].shape[:2] == dummy_trimap.shape
    repr_str = transform.__class__.__name__
    assert repr(transform) == repr_str


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
