# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch


def get_ray_limits_box(rays_o: torch.Tensor, rays_d: torch.Tensor,
                       box_side_length: float
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Author: Petr Kellnhofer
    Intersects rays with the [-1, 1] NDC volume.
    Returns min and max distance of entry.
    Returns -1 for no intersection.
    https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection  # noqa

    Args:
        rays_o (torch.Tensor): The origin of each ray.
        rays_d (torch.Tensor): The direction vector of each ray.
        box_side_length (float): The side length of axis aligned
            bounding box (AABB).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Start and end point
            for each ray. Both shape like (bz, res, res, 1).
    """
    o_shape = rays_o.shape
    rays_o = rays_o.detach().reshape(-1, 3)
    rays_d = rays_d.detach().reshape(-1, 3)

    bb_min = [
        -1 * (box_side_length / 2), -1 * (box_side_length / 2),
        -1 * (box_side_length / 2)
    ]
    bb_max = [
        1 * (box_side_length / 2), 1 * (box_side_length / 2),
        1 * (box_side_length / 2)
    ]
    bounds = torch.tensor([bb_min, bb_max],
                          dtype=rays_o.dtype,
                          device=rays_o.device)
    is_valid = torch.ones(rays_o.shape[:-1], dtype=bool, device=rays_o.device)

    # Precompute inverse for stability.
    invdir = 1 / rays_d
    sign = (invdir < 0).long()

    # Intersect with YZ plane.
    tmin = (bounds.index_select(0, sign[..., 0])[..., 0] -
            rays_o[..., 0]) * invdir[..., 0]
    tmax = (bounds.index_select(0, 1 - sign[..., 0])[..., 0] -
            rays_o[..., 0]) * invdir[..., 0]

    # Intersect with XZ plane.
    tymin = (bounds.index_select(0, sign[..., 1])[..., 1] -
             rays_o[..., 1]) * invdir[..., 1]
    tymax = (bounds.index_select(0, 1 - sign[..., 1])[..., 1] -
             rays_o[..., 1]) * invdir[..., 1]

    # Resolve parallel rays.
    is_valid[torch.logical_or(tmin > tymax, tymin > tmax)] = False

    # Use the shortest intersection.
    tmin = torch.max(tmin, tymin)
    tmax = torch.min(tmax, tymax)

    # Intersect with XY plane.
    tzmin = (bounds.index_select(0, sign[..., 2])[..., 2] -
             rays_o[..., 2]) * invdir[..., 2]
    tzmax = (bounds.index_select(0, 1 - sign[..., 2])[..., 2] -
             rays_o[..., 2]) * invdir[..., 2]

    # Resolve parallel rays.
    is_valid[torch.logical_or(tmin > tzmax, tzmin > tmax)] = False

    # Use the shortest intersection.
    tmin = torch.max(tmin, tzmin)
    tmax = torch.min(tmax, tzmax)

    # Mark invalid.
    tmin[torch.logical_not(is_valid)] = -1
    tmax[torch.logical_not(is_valid)] = -2

    return tmin.reshape(*o_shape[:-1], 1), tmax.reshape(*o_shape[:-1], 1)


def inverse_transform_sampling(bins: torch.Tensor,
                               weights: torch.Tensor,
                               n_importance: int,
                               deterministic: bool = False,
                               eps: float = 1e-5) -> torch.Tensor:
    """Sample `N_importance` samples from `bins` with distribution defined by
    `weights`.

    Args:
        bins (int): (N_points, N_samples+1) where N_samples is the number
            of coarse samples per ray - 2.
        weights (torch.Tensor): Weights shape like (N_points, N_samples-1).
        n_importance (int): The number of samples to draw from the
            distribution.
        deterministic (bool): Whether use deterministic sampling method.
            Defaults to False.
        eps (float): a small number to prevent division by zero.
            Defaults to 1e-5.

    Outputs:
        torch.Tensor: the sampled samples.
    """
    N_rays, N_samples_ = weights.shape
    # prevent division by zero (don't do inplace op!)
    weights = weights + eps
    # (N_rays, N_samples_)
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cumsum(pdf, -1)
    # (N_rays, N_samples_+1)
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)
    # padded to 0~1 inclusive

    if deterministic:
        u = torch.linspace(0, 1, n_importance, device=bins.device)
        u = u.expand(N_rays, n_importance)
    else:
        u = torch.rand(N_rays, n_importance, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds - 1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = torch.stack([below, above],
                               -1).view(N_rays, 2 * n_importance)
    cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, n_importance, 2)
    bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, n_importance, 2)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    # denom equals 0 means a bin has weight 0, in which case it will not
    # be sampled anyway, therefore any value for it is fine (set to 1 here)
    denom[denom < eps] = 1
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def linspace_batch(start: torch.Tensor, stop: torch.Tensor,
                   num: int) -> torch.Tensor:
    """Creates a tensor of shape [num, *start.shape] whose values are evenly
    spaced from start to end, inclusive.

    Replicates but the multi-dimensional behaviour of numpy.linspace in
    PyTorch.

    Args:
        start (torch.Tensor): The start point of each ray. Shape like
            (bz, res, res, 1).
        stop (torch.Tensor): The end point of each ray. Shape like
            (bz, res, res, 1).
        num (int): The number of points to sample.

    Returns:
        torch.Tensor: The sampled points. Shape like (num, bz, res, res, 1)
    """
    # create a tensor of 'num' steps from 0 to 1
    steps = torch.arange(
        num, dtype=torch.float32, device=start.device) / (
            num - 1)

    # reshape the 'steps' tensor to [-1, *([1]*start.ndim)] to allow for
    # broadcastings
    # - using 'steps.reshape([-1, *([1]*start.ndim)])' would be nice here but
    # torchscript cannot statically infer the expected size of a list in this
    # context, hence the code below
    for i in range(start.ndim):
        steps = steps.unsqueeze(-1)

    # the output starts at 'start' and increments until 'stop' in
    # each dimension
    out = start[None] + steps * (stop - start)[None]

    return out
