import cv2
import numpy as np
from mmedit.datasets.pipelines import GenerateTrimap, MergeFgAndBg


def check_keys_contain(result_keys, target_keys):
    """Check if all elements in target_keys is in result_keys."""
    return set(target_keys).issubset(set(result_keys))


def generate_ref_trimap(alpha, kernel_size, iterations, symmetric):
    """Check if a trimap's value is correct."""
    if isinstance(kernel_size, int):
        min_kernel, max_kernel = kernel_size, kernel_size + 1
    else:
        min_kernel, max_kernel = kernel_size

    if isinstance(iterations, int):
        min_iteration, max_iteration = iterations, iterations + 1
    else:
        min_iteration, max_iteration = iterations
    kernel_num = max_kernel - min_kernel

    erode_ksize_idx = np.random.randint(kernel_num)
    erode_ksize = min_kernel + erode_ksize_idx
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                             (erode_ksize, erode_ksize))
    erode_iter = np.random.randint(min_iteration, max_iteration)

    if symmetric:
        dilate_ksize = erode_ksize
        dilate_kernel = erode_kernel
        dilate_iter = erode_iter
    else:
        dilate_ksize_idx = np.random.randint(kernel_num)
        dilate_ksize = min_kernel + dilate_ksize_idx
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                  (dilate_ksize, dilate_ksize))
        dilate_iter = np.random.randint(min_iteration, max_iteration)

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
    target_keys = ['alpha', 'trimap']

    kernel_size = (3, 5)
    iterations = (3, 5)
    symmetric = False
    alpha = np.random.randn(256, 256)
    results = dict(alpha=alpha)
    generate_trimap = GenerateTrimap(kernel_size, iterations, symmetric)
    np.random.seed(123)
    generate_trimap_results = generate_trimap(results)
    trimap = generate_trimap_results['trimap']

    assert check_keys_contain(generate_trimap_results.keys(), target_keys)
    assert trimap.shape == alpha.shape
    np.random.seed(123)
    ref_trimap = generate_ref_trimap(alpha, kernel_size, iterations, symmetric)
    assert (trimap == ref_trimap).all()

    kernel_size = 1
    iterations = 1
    symmetric = True
    generate_trimap = GenerateTrimap(kernel_size, iterations, symmetric)
    np.random.seed(123)
    generate_trimap_results = generate_trimap(results)
    trimap = generate_trimap_results['trimap']

    assert check_keys_contain(generate_trimap_results.keys(), target_keys)
    assert trimap.shape == alpha.shape
    np.random.seed(123)
    ref_trimap = generate_ref_trimap(alpha, kernel_size, iterations, symmetric)
    assert (trimap == ref_trimap).all()

    kernels = [
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (kernel_size, kernel_size))
    ]
    assert repr(generate_trimap) == (
        generate_trimap.__class__.__name__ +
        f'(kernels={kernels}, min_iteration={iterations}, '
        f'max_iteration={iterations + 1}, symmetric={symmetric})')
