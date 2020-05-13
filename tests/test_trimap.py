import cv2
import numpy as np
import pytest
from mmedit.datasets.pipelines import CompositeFg, GenerateTrimap, MergeFgAndBg


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
        src = np.pad(src, pad, constant_values=np.max(src))
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

    fg = np.random.randn(256, 256, 3)
    bg = np.random.randn(256, 256, 3)
    alpha = np.random.randn(256, 256)
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
    alpha = np.random.randn(256, 256)
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


def test_composite_fg():
    target_keys = ['alpha', 'fg', 'bg', 'img_shape']

    np.random.seed(0)
    fg = np.random.rand(240, 320, 3).astype(np.float32)
    bg = np.random.rand(240, 320, 3).astype(np.float32)
    alpha = np.random.rand(240, 320).astype(np.float32)
    results = dict(alpha=alpha, fg=fg, bg=bg, img_shape=(240, 320))
    composite_fg = CompositeFg('tests/data/fg', 'tests/data/alpha', 'jpg',
                               'jpg')
    composite_fg_results = composite_fg(results)
    assert check_keys_contain(composite_fg_results.keys(), target_keys)
    assert composite_fg_results['fg'].shape == (240, 320, 3)

    fg = np.random.rand(240, 320, 3).astype(np.float32)
    bg = np.random.rand(240, 320, 3).astype(np.float32)
    alpha = np.random.rand(240, 320).astype(np.float32)
    results = dict(alpha=alpha, fg=fg, bg=bg, img_shape=(240, 320))
    composite_fg = CompositeFg(
        'tests/data/fg',
        'tests/data/alpha',
        fg_ext='jpg',
        alpha_ext='jpg',
        interpolation='bilinear')
    composite_fg_results = composite_fg(results)
    assert check_keys_contain(composite_fg_results.keys(), target_keys)
    assert composite_fg_results['fg'].shape == (240, 320, 3)

    assert repr(composite_fg) == composite_fg.__class__.__name__ + (
        "(fg_dir='tests/data/fg', alpha_dir='tests/data/alpha', "
        "fg_ext='jpg', alpha_ext='jpg', interpolation='bilinear')")
