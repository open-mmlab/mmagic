# Copyright (c) OpenMMLab. All rights reserved.
import os

import cv2
import numpy as np
import pytest

from mmedit.datasets.pipelines import (CompositeFg, GenerateSeg,
                                       GenerateSoftSeg, GenerateTrimap,
                                       GenerateTrimapWithDistTransform,
                                       MergeFgAndBg, PerturbBg,
                                       TransformTrimap)


def check_keys_contain(result_keys, target_keys):
    """Check if all elements in target_keys is in result_keys."""
    return set(target_keys).issubset(set(result_keys))


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


def test_merge_fg_and_bg():
    target_keys = ['fg', 'bg', 'alpha', 'merged']

    fg = np.random.randn(32, 32, 3)
    bg = np.random.randn(32, 32, 3)
    alpha = np.random.randn(32, 32)
    results = dict(fg=fg, bg=bg, alpha=alpha)
    merge_fg_and_bg = MergeFgAndBg()
    merge_fg_and_bg_results = merge_fg_and_bg(results)

    assert check_keys_contain(merge_fg_and_bg_results.keys(), target_keys)
    assert merge_fg_and_bg_results['merged'].shape == fg.shape


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

    assert check_keys_contain(generate_trimap_results.keys(), target_keys)
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

    assert check_keys_contain(generate_trimap_results.keys(), target_keys)
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
    assert check_keys_contain(generate_trimap_results.keys(), target_keys)
    assert trimap.shape == alpha.shape

    alpha = np.random.randint(0, 256, (32, 32))
    results = dict(alpha=alpha)
    generate_trimap = GenerateTrimapWithDistTransform(dist_thr=3, random=True)
    generate_trimap_results = generate_trimap(results)
    trimap = generate_trimap_results['trimap']
    assert check_keys_contain(generate_trimap_results.keys(), target_keys)
    assert trimap.shape == alpha.shape

    assert repr(generate_trimap) == (
        generate_trimap.__class__.__name__ + '(dist_thr=3, random=True)')


def test_composite_fg():
    target_keys = ['alpha', 'fg', 'bg']

    np.random.seed(0)
    fg = np.random.rand(32, 32, 3).astype(np.float32)
    bg = np.random.rand(32, 32, 3).astype(np.float32)
    alpha = np.random.rand(32, 32).astype(np.float32)
    results = dict(alpha=alpha, fg=fg, bg=bg)
    # use merged dir as fake fg dir, trimap dir as fake alpha dir for unittest
    composite_fg = CompositeFg([
        f'tests{os.sep}data{os.sep}fg', f'tests{os.sep}data{os.sep}merged'
    ], [f'tests{os.sep}data{os.sep}alpha', f'tests{os.sep}data{os.sep}trimap'])
    correct_fg_list = [
        f'tests{os.sep}data{os.sep}fg{os.sep}GT05.jpg',
        f'tests{os.sep}data{os.sep}merged{os.sep}GT05.jpg'
    ]
    correct_alpha_list = [
        f'tests{os.sep}data{os.sep}alpha{os.sep}GT05.jpg',
        f'tests{os.sep}data{os.sep}trimap{os.sep}GT05.png'
    ]
    assert composite_fg.fg_list == correct_fg_list
    assert composite_fg.alpha_list == correct_alpha_list
    composite_fg_results = composite_fg(results)
    assert check_keys_contain(composite_fg_results.keys(), target_keys)
    assert composite_fg_results['fg'].shape == (32, 32, 3)

    fg = np.random.rand(32, 32, 3).astype(np.float32)
    bg = np.random.rand(32, 32, 3).astype(np.float32)
    alpha = np.random.rand(32, 32).astype(np.float32)
    results = dict(alpha=alpha, fg=fg, bg=bg)
    composite_fg = CompositeFg(
        f'tests{os.sep}data{os.sep}fg',
        f'tests{os.sep}data{os.sep}alpha',
        interpolation='bilinear')
    composite_fg_results = composite_fg(results)
    assert check_keys_contain(composite_fg_results.keys(), target_keys)
    assert composite_fg_results['fg'].shape == (32, 32, 3)

    strs = (f"(fg_dirs=['tests{os.sep}data{os.sep}fg'], "
            f"alpha_dirs=['tests{os.sep}data{os.sep}alpha'], "
            "interpolation='bilinear')")
    assert repr(composite_fg) == composite_fg.__class__.__name__ + \
        strs.replace('\\', '\\\\')


def test_generate_seg():
    with pytest.raises(ValueError):
        # crop area should not exceed the image size
        img = np.random.rand(32, 32, 3)
        GenerateSeg._crop_hole(img, (0, 0), (64, 64))

    target_keys = ['alpha', 'trimap', 'seg', 'num_holes']
    alpha = np.random.randint(0, 255, (32, 32))
    trimap = np.zeros_like(alpha)
    trimap[(alpha > 0) & (alpha < 255)] = 128
    trimap[alpha == 255] = 255
    results = dict(alpha=alpha, trimap=trimap)
    generate_seg = GenerateSeg()
    generate_seg_results = generate_seg(results)
    assert check_keys_contain(generate_seg_results.keys(), target_keys)
    assert generate_seg_results['seg'].shape == alpha.shape
    assert isinstance(generate_seg_results['num_holes'], int)
    assert generate_seg_results['num_holes'] < 3

    # check repr string and the default setting
    assert repr(generate_seg) == generate_seg.__class__.__name__ + (
        '(kernel_size=5, erode_iter_range=(10, 20), '
        'dilate_iter_range=(15, 30), num_holes_range=(0, 3), '
        'hole_sizes=[(15, 15), (25, 25), (35, 35), (45, 45)], '
        'blur_ksizes=[(21, 21), (31, 31), (41, 41)]')


def test_perturb_bg():
    with pytest.raises(ValueError):
        # gammma_ratio must be a float between [0, 1]
        PerturbBg(-0.5)

    with pytest.raises(ValueError):
        # gammma_ratio must be a float between [0, 1]
        PerturbBg(1.1)

    target_keys = ['bg', 'noisy_bg']
    # set a random seed to make sure the test goes through every branch
    np.random.seed(123)

    img_shape = (32, 32, 3)
    results = dict(bg=np.random.randint(0, 255, img_shape))
    perturb_bg = PerturbBg(0.6)
    perturb_bg_results = perturb_bg(results)
    assert check_keys_contain(perturb_bg_results.keys(), target_keys)
    assert perturb_bg_results['noisy_bg'].shape == img_shape

    img_shape = (32, 32, 3)
    results = dict(bg=np.random.randint(0, 255, img_shape))
    perturb_bg = PerturbBg(0.6)
    perturb_bg_results = perturb_bg(results)
    assert check_keys_contain(perturb_bg_results.keys(), target_keys)
    assert perturb_bg_results['noisy_bg'].shape == img_shape

    repr_str = perturb_bg.__class__.__name__ + '(gamma_ratio=0.6)'
    assert repr(perturb_bg) == repr_str


def test_generate_soft_seg():
    with pytest.raises(TypeError):
        # fg_thr must be a float
        GenerateSoftSeg(fg_thr=[0.2])
    with pytest.raises(TypeError):
        # border_width must be an int
        GenerateSoftSeg(border_width=25.)
    with pytest.raises(TypeError):
        # erode_ksize must be an int
        GenerateSoftSeg(erode_ksize=5.)
    with pytest.raises(TypeError):
        # dilate_ksize must be an int
        GenerateSoftSeg(dilate_ksize=5.)
    with pytest.raises(TypeError):
        # erode_iter_range must be a tuple of 2 int
        GenerateSoftSeg(erode_iter_range=(3, 5, 7))
    with pytest.raises(TypeError):
        # dilate_iter_range must be a tuple of 2 int
        GenerateSoftSeg(dilate_iter_range=(3, 5, 7))
    with pytest.raises(TypeError):
        # blur_ksizes must be a list of tuple
        GenerateSoftSeg(blur_ksizes=[21, 21])

    target_keys = ['seg', 'soft_seg']

    seg = np.random.randint(0, 255, (512, 512))
    results = dict(seg=seg)

    generate_soft_seg = GenerateSoftSeg(
        erode_ksize=3,
        dilate_ksize=3,
        erode_iter_range=(1, 2),
        dilate_iter_range=(1, 2),
        blur_ksizes=[(11, 11)])
    generate_soft_seg_results = generate_soft_seg(results)
    assert check_keys_contain(generate_soft_seg_results.keys(), target_keys)
    assert generate_soft_seg_results['soft_seg'].shape == seg.shape

    repr_str = generate_soft_seg.__class__.__name__ + (
        '(fg_thr=0.2, border_width=25, erode_ksize=3, dilate_ksize=3, '
        'erode_iter_range=(1, 2), dilate_iter_range=(1, 2), '
        'blur_ksizes=[(11, 11)])')
    assert repr(generate_soft_seg) == repr_str


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
    assert check_keys_contain(results_transformed.keys(), target_keys)
    assert results_transformed['trimap'].shape == dummy_trimap.shape
    assert results_transformed[
        'transformed_trimap'].shape[:2] == dummy_trimap.shape
    repr_str = transform.__class__.__name__
    assert repr(transform) == repr_str
