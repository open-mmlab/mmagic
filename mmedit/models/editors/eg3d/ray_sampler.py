# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch


def sample_rays(cam2world: torch.Tensor, intrinsics: torch.Tensor,
                resolution: int) -> Tuple[torch.Tensor]:
    """Sample origin and direction vectors of rays with passed camera-to-world
    matrix and intrinsics matrix. Noted that skew coefficient is not considered
    in this function.

    Args:
        cam2world (torch.Tensor): The camera-to-world matrix in homogeneous
            coordinates. Shape like (bz, 4, 4).
        intrinsics (torch.Tensor): The intrinsic matrix. Shape like (bz, 3, 3).
        resolution (int): The expect resolution of the render output.

    Returns:
        Tuple[torch.Tensor]: Origins and view directions for rays. Both shape
            like (bz, resolution^2, 3)
    """
    batch_size, n_points = cam2world.shape[0], resolution**2
    cam_in_world = cam2world[:, :3, 3]
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]

    device = cam2world.device

    # torch.meshgrid has been modified in 1.10.0 (compatibility with previous
    # versions), and will be further modified in 1.12 (Breaking Change)
    if 'indexing' in torch.meshgrid.__code__.co_varnames:
        u, v = torch.meshgrid(
            torch.arange(resolution, dtype=torch.float32, device=device),
            torch.arange(resolution, dtype=torch.float32, device=device),
            indexing='ij')
    else:
        u, v = torch.meshgrid(
            torch.arange(resolution, dtype=torch.float32, device=device),
            torch.arange(resolution, dtype=torch.float32, device=device))
    uv = torch.stack([u, v])
    uv = uv * (1. / resolution) + (0.5 / resolution)
    uv = uv.flip(0).reshape(2, -1).transpose(1, 0)
    uv = uv.unsqueeze(0).repeat(cam2world.shape[0], 1, 1)

    x_cam = uv[:, :, 0].view(batch_size, -1)
    y_cam = uv[:, :, 1].view(batch_size, -1)
    z_cam = torch.ones((batch_size, n_points), device=cam2world.device)

    x_lift = (x_cam - cx.unsqueeze(-1)) / fx.unsqueeze(-1) * z_cam
    y_lift = (y_cam - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z_cam

    points_in_cam = torch.stack(
        (x_lift, y_lift, z_cam, torch.ones_like(z_cam)), dim=-1)

    # camera coordinate to world coordinate
    points_in_world = torch.bmm(cam2world, points_in_cam.permute(0, 2, 1))
    points_in_world = points_in_world.permute(0, 2, 1)[:, :, :3]

    ray_dirs = points_in_world - cam_in_world[:, None, :]
    ray_dirs = torch.nn.functional.normalize(ray_dirs, dim=2)

    ray_origins = cam_in_world.unsqueeze(1).repeat(1, ray_dirs.shape[1], 1)

    return ray_origins, ray_dirs
