# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmagic.models.editors.eg3d.eg3d_utils import (get_ray_limits_box,
                                                   inverse_transform_sampling,
                                                   linspace_batch)


def test_get_ray_limits_box_and_linspace_batch():
    rays_o = torch.rand(2, 4, 4, 3)
    rays_d = torch.rand(2, 4, 4, 3)
    rays_start, rays_end = get_ray_limits_box(rays_o, rays_d, 0.5)
    assert rays_start.shape == (2, 4, 4, 1)
    assert rays_end.shape == (2, 4, 4, 1)

    # linspace
    sampled_depth = linspace_batch(rays_start, rays_end, 5)
    assert sampled_depth.shape == (5, 2, 4, 4, 1)


def test_inverse_transform_sampling():
    bins = torch.randn(10, 11)
    weights = torch.randn(10, 9)
    n_importance = 5
    samples = inverse_transform_sampling(bins, weights, n_importance)
    assert samples.shape == (10, 5)

    samples_det_1 = inverse_transform_sampling(
        bins, weights, n_importance, deterministic=True)
    samples_det_2 = inverse_transform_sampling(
        bins, weights, n_importance, deterministic=True)
    assert samples_det_1.shape == (10, 5)
    assert samples_det_2.shape == (10, 5)
    assert (samples_det_1 == samples_det_2).all()


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
