#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from typing import List, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .cam_utils import compute_pitch_yaw_from_w2c_mat

# Essentially, we only need to the following resolution to offset the half-pixel shift from align_corners = False:
# pixel_coordinates = ((grid * scale + 1) / 2) * res - 0.5
# since for boundary condition:
#   - pixel_coordinates = -1
#   - grid_position = -1
# pixel_coordinates = ((-1 * scale + 1) / 2) * res - 0.5 > 0
# ==> res > 1 / (-1 * scale + 1)
# When scale = 0.95, we only need resolution = 20 pixels.
ALIGN_CORNERS_FALSE_NARROW_SCALE = 0.95


def homography(
    rgba: torch.Tensor,
    dhw: torch.Tensor,
    eye_pos: torch.Tensor,
    ray_dir: torch.Tensor,
    z_dir: torch.Tensor,
    assert_not_out_of_plane: bool = False,
    align_corners: bool = True,
    c2w_mat: torch.Tensor = None,
    sphere_c: np.ndarray = None,
):
    """This function samples RGB values from batched plane's RGB-a.

    Args:
        rgba (torch.Tensor): plane's RGB-a value, [B, C, tex_H, tex_W]
        dhw (torch.Tensor): plane's depth, height, and width, [B, 3]
        eye_pos (torch.Tensor): batched camera's position, [B, 3]
        ray_dir (torch.Tensor): batched ray directions from cameras, [B, 3, img_H, img_W]
        z_dir (torch.Tensor): batched forward directions from cameras, [B, 3]
        assert_not_out_of_plane (bool): if set to be True, we will check whether rays go out of plane.
    """
    assert rgba.ndim == 4 and rgba.shape[1] == 4, f'{rgba.shape}'
    assert dhw.ndim == 2 and dhw.shape[1] == 3, f'{dhw.shape}'
    assert eye_pos.ndim == 2 and eye_pos.shape[1] == 3, f'{eye_pos.shape}'
    assert ray_dir.ndim == 4 and ray_dir.shape[1] == 3, f'{ray_dir.shape}'
    assert z_dir.ndim == 2 and z_dir.shape[1] == 3, f'{z_dir.shape}'

    # print("\nin homo: ", rgba.dtype, dhw.dtype, eye_pos.dtype, ray_dir.dtype, z_dir.dtype, "\n")

    n, _, h, w = ray_dir.shape
    # distance, height, width = dhw[0, :]

    # [B, 1]
    distance = dhw[:, :1]
    # [B, 1, 1]
    height = dhw[:, 1:2].unsqueeze(-1)
    # [B, 1, 1]
    width = dhw[:, 2:3].unsqueeze(-1)

    with torch.no_grad():
        # find step size of rays
        z_eye = eye_pos[:, 2:3]
        z_ray = ray_dir[:, 2:3]

        assert torch.all(
            distance >= z_eye[0]
        ), f'Camera must be placed closer to origin than MPI. {distance}, {eye_pos[0, ...]}'

        z_diff = (distance - z_eye).view(n, 1, 1, 1)
        z_diff = z_diff.expand(n, 1, h, w)
        scale = z_diff / z_ray

        # compute ray-plane intersections
        xyz = eye_pos.view(-1, 3, 1, 1) + ray_dir * scale
        # [B, tex_h, tex_w]
        x, y = xyz[:, 0, :, :], xyz[:, 1, :, :]

        # print("\nx, y: ", x.shape, y.shape, "\n")

        # normalize to [-1, 1] for torch.nn.functional.grid_sample
        if align_corners:
            # NOTE: when align_corners = True, -1 and 1 are for center of boundary pixels.
            # Therefore, -1 and 1 are for valid pixels.
            v = 2 * y / height
            u = 2 * x / width
        else:
            # NOTE: when align corners = False, -1 and 1 are for position out of image boundary.
            # Therefore, we narrow down the range a little bit to avoid boundary issues

            v = 2 * y / height
            u = 2 * x / width

            v[(v >= -1)
              & (v <= 1)] = v[(v >= -1)
                              & (v <= 1)] * ALIGN_CORNERS_FALSE_NARROW_SCALE
            u[(u >= -1)
              & (u <= 1)] = u[(u >= -1)
                              & (u <= 1)] * ALIGN_CORNERS_FALSE_NARROW_SCALE

        grid = torch.stack([u, v], dim=-1)

        if assert_not_out_of_plane:

            try:
                assert torch.min(
                    u
                ) >= -1, f"Ray's U direction goes out of plane at {distance}, min val {torch.min(u)}"
                assert torch.max(
                    u
                ) <= 1, f"Ray's U direction goes out of plane at {distance}, max val {torch.max(u)}"
                assert torch.min(
                    v
                ) >= -1, f"Ray's V direction goes out of plane at {distance}, min val {torch.min(v)}"
                assert torch.max(
                    v
                ) <= 1, f"Ray's V direction goes out of plane at {distance}, max val {torch.max(v)}"
            except:

                print('\npos: ', eye_pos[:4, :])
                print('\ndir: ', ray_dir[:4, :3, 0, 0])
                print('\nu: ', u[0, :4, 0])
                print('\nv: ', v[0, :4, 0], '\n')

                if c2w_mat is not None:
                    yaws, pitches = compute_pitch_yaw_from_w2c_mat(
                        torch.inverse(c2w_mat).cpu(),
                        torch.FloatTensor(sphere_c))
                    print('\nyaws: ', yaws.cpu().numpy().tolist(), '\n')
                    print('\npitches: ', pitches.cpu().numpy().tolist(), '\n')

                import sys
                import traceback

                traceback.print_exc()
                err = sys.exc_info()[0]
                print(err)
                sys.exit(1)

    # source code for grid_sample:
    # - https://github.com/pytorch/pytorch/blob/a9b0a921d592b328e7e80a436ef065dadda5f01b/aten/src/ATen/native/GridSampler.cpp
    # - https://github.com/pytorch/pytorch/blob/a9b0a921d592b328e7e80a436ef065dadda5f01b/aten/src/ATen/native/cpu/GridSamplerKernel.cpp
    # See discussion of:
    # - https://github.com/pytorch/pytorch/issues/20785
    # - https://discuss.pytorch.org/t/what-we-should-use-align-corners-false/22663/9
    rgba = F.grid_sample(
        rgba,
        grid,
        align_corners=align_corners,
        mode='bilinear',
        padding_mode='zeros',
    )

    rgb = rgba[:, :3, :, :]
    alpha = rgba[:, 3:4, :, :]

    # compute linear depth and disparity
    with torch.no_grad():
        dist2depth = torch.einsum('nchw,nc->nhw', ray_dir, z_dir)
        depth = scale * dist2depth.view(n, 1, h, w)
        disp = 1 / depth

    return rgb, disp, alpha


class MPI(nn.Module):

    def __init__(self, align_corners=True):
        super().__init__()
        self._align_corners = align_corners

    def check_shapes(
        self,
        *,
        batch_rgba: torch.Tensor,
        batch_dhw: torch.Tensor,
        batch_ray_dir: List[torch.Tensor],
        batch_eye_pos: List[torch.Tensor],
        batch_z_dir: List[torch.Tensor],
        separate_background: Union[None, torch.Tensor],
    ):
        """differentiable rendering function for MPI.

        Args:
            batch_rgba (torch.Tensor): MPI textures of shape (#mpi, #planes, 4, texture_height, texture_width)
            batch_dhw (torch.Tensor): Spatial distance, height, and width of the textures, of shape (#mpi, #planes, 3).
                Note, this is 3D spatial size instead of size with units of pixels.
            batch_ray_dir (List[torch.Tensor]): Ray directions of shape (minibatch, 3, image_height, image_width)
            batch_eye_pos (List[torch.Tensor]): Eye position of shape (minibatch, 3)
            batch_z_dir (List[torch.Tensor]): Optical axis (z director) of shape (minibatch, 3)
            assert_not_out_of_last_plane (bool): if set to be True, we will check whether rays go out of last plane.
        """
        assert (batch_rgba.ndim == 5) and (
            batch_rgba.shape[2] == 4
        ), f'Expected rgba to be of shape (#mpi, #planes, 4, texture_height, texture_width), but instead got {batch_rgba.shape}'
        assert (batch_rgba[:, :, 3, ...].min() >=
                0) and (batch_rgba[:, :, 3, ...].max() <=
                        1), f'Expected alpha to be within the the range [0, 1]'
        assert (
            (batch_dhw.ndim == 3)
            and (batch_dhw.shape[0] == batch_rgba.shape[0])
            and (batch_dhw.shape[1] == batch_rgba.shape[1])
            and (batch_dhw.shape[2] == 3)
        ), f'Expected dhw to be of shape (#mpi, #planes, 3), but instead got {batch_dhw.shape} (rgba: {batch_rgba.shape})'

        assert len(batch_ray_dir) == batch_rgba.shape[
            0], f'{len(batch_ray_dir)}, {batch_rgba.shape[0]}'
        assert len(batch_eye_pos) == batch_rgba.shape[
            0], f'{len(batch_eye_pos)}, {batch_rgba.shape[0]}'
        assert len(batch_z_dir) == batch_rgba.shape[
            0], f'{len(batch_z_dir)}, {batch_rgba.shape[0]}'

        for i in range(len(batch_ray_dir)):
            assert (batch_ray_dir[i].ndim == 4) and (
                batch_ray_dir[i].shape[1] == 3
            ), (f'Expected ray_dir to be of shape (minibatch, 3, image_height, image_width), '
                f'but instead got {batch_ray_dir[i].shape} for {i} th elem.')
            assert (batch_eye_pos[i].ndim == 2) and (
                batch_eye_pos[i].shape[1] == 3
            ), (f'Expected eye_pos to be of shape (minibatch, 3), '
                f'but instead got {batch_eye_pos[i].shape} for {i} th elem.')
            assert (batch_z_dir[i].ndim == 2) and (
                batch_z_dir[i].shape[1] == 3), (
                    f'Expected z_dir to be of shape (minibatch, 3), '
                    f'but instead got {batch_z_dir[i].shape} for {i} th elem.')

        if separate_background is not None:
            assert separate_background.ndim == 4 and separate_background.shape[
                1] == 3, (f'Expect background to be of shape (#mpi, 3, h, w), '
                          f'but instead get {separate_background.shape}.')

    def forward(
        self,
        *,
        batch_rgba: torch.Tensor,
        batch_dhw: torch.Tensor,
        batch_ray_dir: List[torch.Tensor],
        batch_eye_pos: List[torch.Tensor],
        batch_z_dir: List[torch.Tensor],
        separate_background: Union[None, torch.Tensor],
        assert_not_out_of_last_plane: bool = False,
        c2w_mat: torch.Tensor = None,
        sphere_c: np.ndarray = None,
    ):
        self.check_shapes(
            batch_rgba=batch_rgba,
            batch_dhw=batch_dhw,
            batch_ray_dir=batch_ray_dir,
            batch_eye_pos=batch_eye_pos,
            batch_z_dir=batch_z_dir,
            separate_background=separate_background,
        )

        # expand and concatenate MPIs
        cat_rgbas = []
        cat_dhws = []
        cat_separate_backgrounds = []
        for i in range(len(batch_ray_dir)):
            tmp_n_views = batch_ray_dir[i].shape[0]
            cat_rgbas.append(batch_rgba[i:(i + 1),
                                        ...].expand(tmp_n_views, -1, -1, -1,
                                                    -1))
            cat_dhws.append(batch_dhw[i:(i + 1),
                                      ...].expand(tmp_n_views, -1, -1))
            if separate_background is not None:
                cat_separate_backgrounds.append(
                    separate_background[i:(i + 1),
                                        ...].expand(tmp_n_views, -1, -1, -1))
        # [#mpi x #cameras, #planes, 4, tex_h, tex_w]
        cat_rgbas = torch.cat(cat_rgbas, dim=0)
        # [#mpi x #cameras, #planes, 3]
        cat_dhws = torch.cat(cat_dhws, dim=0)
        if separate_background is not None:
            # [#mpi x #cameras, 1, 3, tex_h, tex_w]
            cat_separate_backgrounds = torch.cat(
                cat_separate_backgrounds, dim=0)

        # concatenate all camera-related information
        # [#mpi x #cameras, 3, img_h, img_w]
        cat_ray_dir = torch.cat(batch_ray_dir, dim=0)
        # [#mpi x #cameras, 3]
        cat_eye_pos = torch.cat(batch_eye_pos, dim=0)
        # [#mpi x #cameras, 3]
        cat_z_dir = torch.cat(batch_z_dir, dim=0)

        num_layers = batch_dhw.shape[1]
        n_total_views, _, img_h, img_w = cat_ray_dir.shape
        _, _, _, tex_h, tex_w = cat_rgbas.shape

        # NOTE: we expand camera-related information by #planes times
        # [#mpi x #cameras, 1, 3, img_h, img_w] -> [#mpi x #cameras, #planes, 3, img_h, img_w] -> [#mpi x #cameras x #planes, 3, img_h, img_w]
        flat_ray_dir = (
            cat_ray_dir.unsqueeze(1).expand(-1, num_layers, -1, -1,
                                            -1).reshape(
                                                (n_total_views * num_layers, 3,
                                                 img_h, img_w)))
        # [#mpi x #cameras, 3] -> # [#mpi x #cameras, 1, 3] -> # [#mpi x #cameras, #planes, 3] -> # [#mpi x #cameras x #planes, 3]
        flat_eye_pos = cat_eye_pos.unsqueeze(1).expand(-1, num_layers,
                                                       -1).reshape(
                                                           (n_total_views *
                                                            num_layers, 3))
        # [#mpi x #cameras, 3] -> # [#mpi x #cameras, 1, 3] -> # [#mpi x #cameras, #planes, 3] -> # [#mpi x #cameras x #planes, 3]
        flat_z_dir = cat_z_dir.unsqueeze(1).expand(-1, num_layers, -1).reshape(
            (n_total_views * num_layers, 3))

        # # NOTE: we need to reverse the planes to make sure furthest plane comes first
        # cat_rgbas = torch.flip(cat_rgbas, [1])
        # cat_dhws = torch.flip(cat_dhws, [1])

        # [#mpi x #cameras, #planes, 4, tex_h, tex_w] -> [#mpi x #cameras x #planes, 4, tex_h, tex_w]
        flat_rgbas = cat_rgbas.reshape(
            (n_total_views * num_layers, 4, tex_h, tex_w))
        # [#mpi x #cameras, #planes, 3] -> # [#mpi x #cameras x #planes, 3]
        flat_dhws = cat_dhws.reshape((n_total_views * num_layers, 3))

        if assert_not_out_of_last_plane:
            # this will ensure that all rays have interaction with at least one plane of MPI
            with torch.no_grad():
                # print("\nCheck: ", flat_dhws[(num_layers-1)::num_layers, ...], "\n")
                _ = homography(
                    flat_rgbas[(num_layers - 1)::num_layers, ...],
                    flat_dhws[(num_layers - 1)::num_layers, ...],
                    flat_eye_pos[(num_layers - 1)::num_layers, ...],
                    flat_ray_dir[(num_layers - 1)::num_layers, ...],
                    flat_z_dir[(num_layers - 1)::num_layers, ...],
                    assert_not_out_of_plane=True,
                    align_corners=self._align_corners,
                    c2w_mat=c2w_mat,
                    sphere_c=sphere_c,
                )

        # get rgba image inverse-warped from texture to camera
        # [#mpi x #cameras x #planes, 3, img_h, img_w]
        flat_render_rgb, flat_render_disp, flat_render_alpha = homography(
            flat_rgbas,
            flat_dhws,
            flat_eye_pos,
            flat_ray_dir,
            flat_z_dir,
            assert_not_out_of_plane=False,
            align_corners=self._align_corners,
            c2w_mat=c2w_mat,
            sphere_c=sphere_c,
        )

        flat_render_depth = 1 / flat_render_disp

        # NOTE: the 1st plane is the closest one
        cat_render_alpha = flat_render_alpha.reshape(
            (n_total_views, num_layers, 1, img_h, img_w))
        cat_render_rgb = flat_render_rgb.reshape(
            (n_total_views, num_layers, 3, img_h, img_w))
        cat_render_disp = flat_render_disp.reshape(
            (n_total_views, num_layers, 1, img_h, img_w))
        cat_render_depth = flat_render_depth.reshape(
            (n_total_views, num_layers, 1, img_h, img_w))

        # alpha-composition
        # [#mpi x #cameras, #planes + 1, 1, img_h, img_w]
        alphas_shifted = torch.cat([
            torch.ones_like(cat_render_alpha[:, :1, ...]),
            1 - cat_render_alpha + 1e-10
        ], 1)
        # [#mpi x #cameras, #planes, 1, img_h, img_w]
        weights = cat_render_alpha * torch.cumprod(
            alphas_shifted, dim=1)[:, :-1, ...]

        weights_sum = weights.sum(1)
        # if last_back:
        #     weights[:, -1, ...] += (1 - weights_sum)

        # [#mpi x #cameras, 3, img_h, img_w]
        color_out = torch.sum(weights * cat_render_rgb, dim=1)
        # [#mpi x #cameras, 1, img_h, img_w]
        disp_out = torch.sum(weights * cat_render_disp, dim=1)
        # [#mpi x #cameras, 1, img_h, img_w]
        depth_out = torch.sum(weights * cat_render_depth, dim=1)

        return color_out, depth_out
