# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path

import numpy as np
import pytest
from mmengine.fileio import load

from mmagic.datasets.transforms import (CompositeFg, MergeFgAndBg, PerturbBg,
                                        RandomJitter, RandomLoadResizeBg)

test_root = Path(__file__).parent.parent.parent
data_root = test_root / 'data' / 'matting_dataset'


def assert_keys_contain(result_keys, target_keys):
    """Check if all elements in target_keys is in result_keys."""
    assert set(target_keys).issubset(set(result_keys))


def test_composite_fg():
    target_keys = ['alpha', 'fg', 'bg']

    fg = np.random.rand(32, 32, 3).astype(np.float32)
    bg = np.random.rand(32, 32, 3).astype(np.float32)
    alpha = np.random.rand(32, 32, 1).astype(np.float32)
    results = dict(alpha=alpha, fg=fg, bg=bg)
    # use merged dir as fake fg dir, trimap dir as fake alpha dir for unittest
    composite_fg = CompositeFg(
        [str(data_root / 'fg'),
         str(data_root / 'merged')],
        [str(data_root / 'alpha'),
         str(data_root / 'trimap')])
    correct_fg_list = [
        str(data_root / 'fg' / 'GT05.jpg'),
        str(data_root / 'merged' / 'GT05.jpg')
    ]
    correct_alpha_list = [
        str(data_root / 'alpha' / 'GT05.jpg'),
        str(data_root / 'trimap' / 'GT05.png')
    ]
    assert composite_fg.fg_list == correct_fg_list
    assert composite_fg.alpha_list == correct_alpha_list
    for _ in range(3):  # to test randomness
        composite_fg_results = composite_fg(results)
        assert_keys_contain(composite_fg_results.keys(), target_keys)
        assert composite_fg_results['fg'].shape == (32, 32, 3)
        assert composite_fg_results['alpha'].shape == (32, 32, 1)

    fg = np.random.rand(32, 32, 3).astype(np.float32)
    bg = np.random.rand(32, 32, 3).astype(np.float32)
    alpha = np.random.rand(32, 32, 1).astype(np.float32)
    results = dict(alpha=alpha, fg=fg, bg=bg)
    composite_fg = CompositeFg(
        str(data_root / 'fg'),
        str(data_root / 'alpha'),
        interpolation='bilinear')
    for _ in range(3):  # to test randomness
        composite_fg_results = composite_fg(results)
        assert_keys_contain(composite_fg_results.keys(), target_keys)
        assert composite_fg_results['fg'].shape == (32, 32, 3)
        assert composite_fg_results['alpha'].shape == (32, 32, 1)

    _fg_dirs = str(data_root / 'fg')
    _alpha_dirs = str(data_root / 'alpha')
    strs = (f"(fg_dirs=['{_fg_dirs}'], "
            f"alpha_dirs=['{_alpha_dirs}'], "
            "interpolation='bilinear')")

    assert repr(composite_fg) == 'CompositeFg' + strs.replace('\\', '\\\\')


def test_merge_fg_and_bg():
    target_keys = ['fg', 'bg', 'alpha', 'merged']

    fg = np.random.randn(32, 32, 3)
    bg = np.random.randn(32, 32, 3)
    alpha = np.random.randn(32, 32, 1)
    results = dict(fg=fg, bg=bg, alpha=alpha)
    merge_fg_and_bg = MergeFgAndBg()
    merge_fg_and_bg_results = merge_fg_and_bg(results)

    assert_keys_contain(merge_fg_and_bg_results.keys(), target_keys)
    assert merge_fg_and_bg_results['merged'].shape == fg.shape
    assert merge_fg_and_bg_results['alpha'].shape == (32, 32, 1)


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
    assert_keys_contain(perturb_bg_results.keys(), target_keys)
    assert perturb_bg_results['noisy_bg'].shape == img_shape

    img_shape = (32, 32, 3)
    results = dict(bg=np.random.randint(0, 255, img_shape))
    perturb_bg = PerturbBg(0.6)
    perturb_bg_results = perturb_bg(results)
    assert_keys_contain(perturb_bg_results.keys(), target_keys)
    assert perturb_bg_results['noisy_bg'].shape == img_shape

    repr_str = perturb_bg.__class__.__name__ + '(gamma_ratio=0.6)'
    assert repr(perturb_bg) == repr_str


def test_random_jitter():
    with pytest.raises(AssertionError):
        RandomJitter(-40)

    with pytest.raises(AssertionError):
        RandomJitter((-40, 40, 40))

    target_keys = ['fg']

    fg = np.random.rand(240, 320, 3).astype(np.float32)
    alpha = np.random.rand(240, 320, 1).astype(np.float32)
    results = dict(fg=fg.copy(), alpha=alpha)
    random_jitter = RandomJitter(40)
    random_jitter_results = random_jitter(results)
    assert_keys_contain(random_jitter_results.keys(), target_keys)
    assert random_jitter_results['fg'].shape == (240, 320, 3)

    fg = np.random.rand(240, 320, 3).astype(np.float32)
    alpha = np.random.rand(240, 320, 1).astype(np.float32)
    results = dict(fg=fg.copy(), alpha=alpha)
    random_jitter = RandomJitter((-50, 50))
    random_jitter_results = random_jitter(results)
    assert_keys_contain(random_jitter_results.keys(), target_keys)
    assert random_jitter_results['fg'].shape == (240, 320, 3)

    assert repr(random_jitter) == random_jitter.__class__.__name__ + (
        'hue_range=(-50, 50)')


def test_random_load_resize_bg():
    ann_file = data_root / 'ann_old.json'
    bg_dir = data_root / 'bg'
    data_infos = load(ann_file)
    results = dict()
    for data_info in data_infos:
        for key in data_info:
            results[key] = str(data_root / data_info[key])
    target_keys = ['bg']

    results = dict(fg=np.random.rand(128, 128))
    random_load_bg = RandomLoadResizeBg(bg_dir=bg_dir)
    for _ in range(2):
        random_load_bg_results = random_load_bg(results)
        """Check if all elements in target_keys is in result_keys."""
        assert set(target_keys).issubset(set(random_load_bg_results.keys()))
        assert isinstance(random_load_bg_results['bg'], np.ndarray)
        assert random_load_bg_results['bg'].shape == (128, 128, 3)

    assert repr(
        random_load_bg) == f"RandomLoadResizeBg(bg_dir='{str(bg_dir)}')"


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
