# Copyright (c) OpenMMLab. All rights reserved.

import numpy as np
import pytest

from mmagic.datasets.transforms import GenerateSeg, GenerateSoftSeg


def assert_keys_contain(result_keys, target_keys):
    """Check if all elements in target_keys is in result_keys."""
    assert set(target_keys).issubset(set(result_keys))


def test_generate_seg():
    with pytest.raises(ValueError):
        # crop area should not exceed the image size
        img = np.random.rand(32, 32, 3)
        GenerateSeg._crop_hole(img, (0, 0), (64, 64))

    target_keys = ['alpha', 'trimap', 'seg', 'num_holes']
    alpha = np.random.randint(0, 255, (64, 64))
    # previously this is 32 by 32, but default hole size contains 45
    trimap = np.zeros_like(alpha)
    trimap[(alpha > 0) & (alpha < 255)] = 128
    trimap[alpha == 255] = 255
    results = dict(alpha=alpha, trimap=trimap)
    for _ in range(3):
        # test multiple times as it contain some random operations
        generate_seg = GenerateSeg(num_holes_range=(1, 3))
        generate_seg_results = generate_seg(results)
        assert_keys_contain(generate_seg_results.keys(), target_keys)
        assert generate_seg_results['seg'].shape == alpha.shape
        assert isinstance(generate_seg_results['num_holes'], int)
        assert generate_seg_results['num_holes'] < 3

    # check repr string and the default setting
    assert repr(generate_seg) == generate_seg.__class__.__name__ + (
        '(kernel_size=5, erode_iter_range=(10, 20), '
        'dilate_iter_range=(15, 30), num_holes_range=(1, 3), '
        'hole_sizes=[(15, 15), (25, 25), (35, 35), (45, 45)], '
        'blur_ksizes=[(21, 21), (31, 31), (41, 41)]')


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
    assert_keys_contain(generate_soft_seg_results.keys(), target_keys)
    assert generate_soft_seg_results['soft_seg'].shape == seg.shape

    repr_str = generate_soft_seg.__class__.__name__ + (
        '(fg_thr=0.2, border_width=25, erode_ksize=3, dilate_ksize=3, '
        'erode_iter_range=(1, 2), dilate_iter_range=(1, 2), '
        'blur_ksizes=[(11, 11)])')
    assert repr(generate_soft_seg) == repr_str


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
