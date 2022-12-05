# Copyright (c) OpenMMLab. All rights reserved.
"""The renderer is a module that takes in rays, decides where to sample along
each ray, and computes pixel colors using the volume rendering equation."""

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION

from ..stylegan3.stylegan3_modules import FullyConnectedLayer
from .eg3d_utils import (get_ray_limits_box, inverse_transform_sampling,
                         linspace_batch)


class EG3DRenderer(BaseModule):
    """Renderer for EG3D. This class samples render points on each input ray
    and interpolate the triplane feature corresponding to the points'
    coordinates. Then, predict each point's RGB feature and density (sigma) by
    a neural network and calculate the RGB feature of each ray by integration.
    Different from typical NeRF models, the decoder of EG3DRenderer takes
    triplane feature of each points as input instead of positional encoding of
    the coordinates.

    Args:
        decoder_cfg (dict): The config to build neural renderer.
        ray_start (float): The start position of all rays.
        ray_end (float): The end position of all rays.
        box_warp (float): The side length of the cube spanned by the triplanes.
            The box is axis-aligned, centered at the origin. The range of each
            axis is `[-box_warp/2, box_warp/2]`. If `box_warp=1.8`, it has
            vertices at the range of axis is `[-0.9, 0.9]`. Defaults to 1.
        depth_resolution (int): Resolution of depth, as well as the number of
            points per ray. Defaults to 64.
        depth_resolution_importance (int): Resolution of depth in hierarchical
            sampling. Defaults to 64.
        clamp_mode (str): The clamp mode for density predicted by nerural
            renderer. Defaults to 'softplus'.
        white_back (bool): Whether render a white background. Defaults to True.
        projection_mode (str): The projection method to mapping coordinates of
            render points to plane feature. The usage of this argument please
            refer to :meth:`self.project_onto_planes` and
            https://github.com/NVlabs/eg3d/issues/67. Defaults to 'Official'.
    """

    def __init__(self,
                 decoder_cfg: dict,
                 ray_start: float,
                 ray_end: float,
                 box_warp: float = 1,
                 depth_resolution: int = 64,
                 depth_resolution_importance: int = 64,
                 density_noise: float = 0,
                 clamp_mode: str = 'softplus',
                 white_back: bool = True,
                 projection_mode: str = 'Official'):
        super().__init__()
        self.decoder = EG3DDecoder(**decoder_cfg)

        self.ray_start = ray_start
        self.ray_end = ray_end
        self.box_warp = box_warp
        self.depth_resolution = depth_resolution
        self.depth_resolution_importance = depth_resolution_importance
        self.density_noise = density_noise

        self.clamp_mode = clamp_mode
        self.white_back = white_back
        self.projection_mode = projection_mode

    def get_value(self,
                  target: str,
                  render_kwargs: Optional[dict] = None) -> Any:
        """Get value of target field.

        Args:
            target (str): The key of the target field.
            render_kwargs (Optional[dict], optional): The input key word
                arguments dict. Defaults to None.

        Returns:
            Any: The default value of target field.
        """
        if render_kwargs is None:
            return getattr(self, target)
        return render_kwargs.get(target, getattr(self, target))

    def forward(self,
                planes: torch.Tensor,
                ray_origins: torch.Tensor,
                ray_directions: torch.Tensor,
                render_kwargs: Optional[dict] = None) -> Tuple[torch.Tensor]:
        """Render 2D RGB feature, weighed depth and weights with the passed
        triplane features and rays. 'weights' denotes `w` in Equation 5 of the
        NeRF's paper.

        Args:
            planes (torch.Tensor): The triplane features shape like
                (bz, 3, TriPlane_feat, TriPlane_res, TriPlane_res).
            ray_origins (torch.Tensor): The original of each ray to render,
                shape like (bz, NeRF_res * NeRF_res, 3).
            ray_directions (torch.Tensor): The direction vector of each ray to
                render, shape like (bz, NeRF_res * NeRF_res, 3).
            render_kwargs (Optional[dict], optional): The specific kwargs for
                rendering. Defaults to None.

        Returns:
            Tuple[torch.Tensor]: Renderer RGB feature, weighted depths and
                weights.
        """
        ray_start = self.get_value('ray_start', render_kwargs)
        ray_end = self.get_value('ray_end', render_kwargs)
        box_warp = self.get_value('box_warp', render_kwargs)
        depth_resolution = self.get_value('depth_resolution', render_kwargs)
        depth_resolution_importance = self.get_value(
            'depth_resolution_importance', render_kwargs)
        density_noise = self.get_value('density_noise', render_kwargs)

        if ray_start == ray_end == 'auto':
            ray_start, ray_end = get_ray_limits_box(
                ray_origins, ray_directions, box_side_length=box_warp)
            is_ray_valid = ray_end > ray_start
            if torch.any(is_ray_valid).item():
                ray_start[~is_ray_valid] = ray_start[is_ray_valid].min()
                ray_end[~is_ray_valid] = ray_start[is_ray_valid].max()
            depths_coarse = self.sample_stratified(ray_origins, ray_start,
                                                   ray_end, depth_resolution)
        else:
            assert (isinstance(ray_start, float) and isinstance(
                ray_end, float)), (
                    '\'ray_start\' and \'ray_end\' must be both float type or '
                    f'both \'auto\'. But receive {ray_start} and {ray_end}.')
            assert ray_start < ray_end, (
                '\'ray_start\' must less than \'ray_end\'.')
            # Create stratified depth samples
            depths_coarse = self.sample_stratified(ray_origins, ray_start,
                                                   ray_end, depth_resolution)

        batch_size, num_rays, samples_per_ray, _ = depths_coarse.shape

        # Coarse Pass
        sample_coordinates = (
            ray_origins.unsqueeze(-2) +
            depths_coarse * ray_directions.unsqueeze(-2)).reshape(
                batch_size, -1, 3)

        out = self.neural_rendering(planes, sample_coordinates, density_noise,
                                    box_warp)
        colors_coarse = out['rgb']
        densities_coarse = out['sigma']
        colors_coarse = colors_coarse.reshape(batch_size, num_rays,
                                              samples_per_ray,
                                              colors_coarse.shape[-1])
        densities_coarse = densities_coarse.reshape(batch_size, num_rays,
                                                    samples_per_ray, 1)

        # Fine Pass
        N_importance = depth_resolution_importance
        if N_importance is not None and N_importance > 0:
            _, _, weights = self.volume_rendering(colors_coarse,
                                                  densities_coarse,
                                                  depths_coarse)

            depths_fine = self.sample_importance(depths_coarse, weights,
                                                 N_importance)
            sample_coordinates = (
                ray_origins.unsqueeze(-2) +
                depths_fine * ray_directions.unsqueeze(-2)).reshape(
                    batch_size, -1, 3)

            out = self.neural_rendering(planes, sample_coordinates,
                                        density_noise, box_warp)
            colors_fine = out['rgb']
            densities_fine = out['sigma']
            colors_fine = colors_fine.reshape(batch_size, num_rays,
                                              N_importance,
                                              colors_fine.shape[-1])
            densities_fine = densities_fine.reshape(batch_size, num_rays,
                                                    N_importance, 1)

            all_depths, all_colors, all_densities = self.unify_samples(
                depths_coarse, colors_coarse, densities_coarse, depths_fine,
                colors_fine, densities_fine)

            # Aggregate
            rgb_final, depth_final, weights = self.volume_rendering(
                all_colors, all_densities, all_depths)
        else:
            rgb_final, depth_final, weights = self.volume_rendering(
                colors_coarse, densities_coarse, depths_coarse)

        return rgb_final, depth_final, weights.sum(2)

    def sample_stratified(self, ray_origins: torch.Tensor,
                          ray_start: Union[float, torch.Tensor],
                          ray_end: Union[float, torch.Tensor],
                          depth_resolution: int) -> torch.Tensor:
        """Return depths of approximately uniformly spaced samples along rays.

        Args:
            ray_origins (torch.Tensor): The original of each ray, shape like
                (bz, NeRF_res * NeRF_res, 3). Only used to provide
                device and shape info.
            ray_start (Union[float, torch.Tensor]): The start position of rays.
                If a float is passed, all rays will have the same start
                distance.
            ray_end (Union[float, torch.Tensor]): The end position of rays. If
                a float is passed, all rays will have the same end distance.
            depth_resolution (int): Resolution of depth, as well as the number
                of points per ray.

        Returns:
            torch.Tensor: The sampled coarse depth shape like
                (bz, NeRF_res * NeRF_res, 1).
        """
        N, M, _ = ray_origins.shape
        if isinstance(ray_start, torch.Tensor):
            # perform linspace for batch of tensor
            depths_coarse = linspace_batch(ray_start, ray_end,
                                           depth_resolution)
            depths_coarse = depths_coarse.permute(1, 2, 0, 3)
            depth_delta = (ray_end - ray_start) / (depth_resolution - 1)
            depths_coarse += torch.rand_like(depths_coarse) * depth_delta[...,
                                                                          None]
        else:
            depths_coarse = torch.linspace(
                ray_start,
                ray_end,
                depth_resolution,
                device=ray_origins.device)
            depths_coarse = depths_coarse.reshape(1, 1, depth_resolution, 1)
            depths_coarse = depths_coarse.repeat(N, M, 1, 1)

            depth_delta = (ray_end - ray_start) / (depth_resolution - 1)
            depths_coarse += torch.rand_like(depths_coarse) * depth_delta

        return depths_coarse

    def neural_rendering(self, planes: torch.Tensor, sample_coordinates: float,
                         density_noise: float, box_warp: float) -> dict:
        """Predict RGB features and densities of the coordinates by neural
        renderer model and the triplane input.

        Args:
            planes (torch.Tensor): Triplane feature shape like
                (bz, 3, TriPlane_feat, TriPlane_res, TriPlane_res).
            sample_coordinates (torch.Tensor): Coordinates of the sampling
                points, shape like (bz, N_depth * NeRF_res * NeRF_res, 1).
            density_noise (float): Strength of noise add to the predicted
                density.
            box_warp (float): The side length of the cube spanned by the
                triplanes.

        Returns:
            dict: A dict contains RGB features ('rgb') and densities ('sigma').
        """
        sampled_features = self.sample_from_planes(
            planes, sample_coordinates, box_warp=box_warp)

        out = self.decoder(sampled_features)
        if density_noise > 0:
            out['sigma'] += torch.randn_like(out['sigma']) * density_noise
        return out

    def sample_from_planes(self,
                           plane_features: torch.Tensor,
                           coordinates: torch.Tensor,
                           interp_mode: str = 'bilinear',
                           box_warp: float = None) -> torch.Tensor:
        """Sample from feature from triplane feature with the passed
        coordinates of render points.

        Args:
            plane_features (torch.Tensor): The triplane feature.
            coordinates (torch.Tensor): The coordinates of points to render.
            interp_mode (str): The interpolation mode to sample feature from
                triplane.
            box_warp (float): The side length of the cube spanned by the
                triplanes.

        Returns:
            torch.Tensor: The sampled triplane feature of the render points.
        """
        N, n_planes, C, H, W = plane_features.shape
        _, M, _ = coordinates.shape
        plane_features = plane_features.view(N * n_planes, C, H, W)

        coordinates = (2 / box_warp) * coordinates
        # NOTE: do not support change projection_mode for specific renderer,
        # use self.projection_mode
        projected_coordinates = self.project_onto_planes(coordinates)
        projected_coordinates = projected_coordinates[:, None, ...]

        output_features = torch.nn.functional.grid_sample(
            plane_features,
            projected_coordinates.float(),
            mode=interp_mode,
            padding_mode='zeros',
            align_corners=False)
        output_features = output_features.permute(0, 3, 2, 1).reshape(
            N, n_planes, M, C)
        return output_features

    def project_onto_planes(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Project 3D points to plane formed by coordinate axes. In this
        function, we use indexing operation to replace matrix multiplication to
        achieve higher calculation performance.

        In the original implementation, the mapping matrix is incorrect.
        Therefore we support users to define `projection_mode` to control
        projection behavior in the initialization function of
        :class:~`EG3DRenderer`. If you want to run inference with the offifical
        pretrained model, please remember to set
        `projection_mode = 'official'`. More information please refer to
        https://github.com/NVlabs/eg3d/issues/67.

        If the project mode `official`, the equivalent projection matrix is
        inverse matrix of:

            [[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
            [[0, 0, 1], [1, 0, 0], [0, 1, 0]]]

        Otherwise, the equivalent projection matrix is inverse matrix of:

            [[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
            [[0, 0, 1], [0, 1, 0], [1, 0, 0]]]

        Args:
            coordinates (torch.Tensor): The coordinates of the render points.
                shape like (bz, NeRF_res * NeRF_res * N_depth, 3).

        Returns:
            torch.Tensor: The projected coordinates.
        """
        N, _, _ = coordinates.shape
        xy_coord = coordinates[:, :, (0, 1)]  # (bz, N_points, 3)
        xz_coord = coordinates[:, :, (0, 2)]  # (bz, N_points, 3)
        if self.projection_mode.upper() == 'OFFICIAL':
            yz_coord = coordinates[:, :, (2, 0)]  # actually zx_coord
        else:
            yz_coord = coordinates[:, :, (2, 1)]
        coord_proejcted = torch.cat([xy_coord, xz_coord, yz_coord], dim=0)
        # create a index list to release the following remapping:
        # [xy, xy, ..., xz, xz, ..., yz, yz, ...] -> [xy, xz, yz, ...]
        index = []
        for n in range(N):
            index += [n, N + n, N * 2 + n]
        return coord_proejcted[index, ...]

    def unify_samples(self, depths_c: torch.Tensor, colors_c: torch.Tensor,
                      densities_c: torch.Tensor, depths_f: torch.Tensor,
                      colors_f: torch.Tensor,
                      densities_f: torch.Tensor) -> Tuple[torch.Tensor]:
        """Sort and merge coarse samples and fine samples.

        Args:
            depths_c (torch.Tensor): Coarse depths shape like
                (bz, NeRF_res * NeRF_res, N_depth, 1).
            colors_c (torch.Tensor): Coarse color features shape like
                (bz, NeRF_res * NeRF_res, N_depth, N_feat).
            densities_c (torch.Tensor): Coarse densities shape like
                (bz, NeRF_res * NeRF_res, N_depth, 1).
            depths_f (torch.Tensro): Fine depths shape like
                (bz, NeRF_res * NeRF_res, N_depth_fine, 1).
            colors_f (torch.Tensor): Fine colors features shape like
                (bz, NeRF_res * NeRF_res, N_depth_fine, N_feat).
            densities_f (torch.Tensor): Fine densites shape like
                (bz, NeRF_res * NeRF_res, N_depth_fine, 1).

        Returns:
            Tuple[torch.Tensor]: Unified depths, color features and densities.
                The third dimension of returns are `N_depth + N_depth_fine`.
        """
        all_depths = torch.cat([depths_c, depths_f], dim=-2)
        all_colors = torch.cat([colors_c, colors_f], dim=-2)
        all_densities = torch.cat([densities_c, densities_f], dim=-2)

        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(
            all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_densities = torch.gather(all_densities, -2,
                                     indices.expand(-1, -1, -1, 1))

        return all_depths, all_colors, all_densities

    def volume_rendering(self, colors: torch.Tensor, densities: torch.Tensor,
                         depths: torch.Tensor) -> Tuple[torch.Tensor]:
        """Volume rendering.

        Args:
            colors (torch.Tensor): Color feature for each points. Shape like
                (bz, N_points, N_depth, N_feature).
            densities (torch.Tensor): Density for each points. Shape like
                (bz, N_points, N_depth, 1).
            depths (torch.Tensor): Depths for each points. Shape like
                (bz, N_points, N_depth, 1).

        Returns:
            Tuple[torch.Tensor]: A tuple of color feature
                `(bz, N_points, N_feature)`, weighted depth
                `(bz, N_points, 1)` and weight
                `(bz, N_points, N_depth-1, 1)`.
        """
        deltas = depths[:, :, 1:] - depths[:, :, :-1]
        colors_mid = (colors[:, :, :-1] + colors[:, :, 1:]) / 2
        densities_mid = (densities[:, :, :-1] + densities[:, :, 1:]) / 2
        depths_mid = (depths[:, :, :-1] + depths[:, :, 1:]) / 2

        if self.clamp_mode == 'softplus':
            # activation bias of -1 makes things initialize better
            densities_mid = F.softplus(densities_mid - 1)
        else:
            assert False, (
                'EG3DRenderer only supports \'softplus\' for \'clamp_mode\', '
                f'but receive \'{self.clamp_mode}\'.')

        density_delta = densities_mid * deltas

        alpha = 1 - torch.exp(-density_delta)

        alpha_shifted = torch.cat(
            [torch.ones_like(alpha[:, :, :1]), 1 - alpha + 1e-10], -2)
        weights = alpha * torch.cumprod(alpha_shifted, -2)[:, :, :-1]

        composite_rgb = torch.sum(weights * colors_mid, -2)
        weight_total = weights.sum(2)
        composite_depth = torch.sum(weights * depths_mid, -2) / weight_total

        # clip the composite to min/max range of depths
        if digit_version(TORCH_VERSION) < digit_version('1.8.0'):
            composite_depth[torch.isnan(composite_depth)] = float('inf')
        else:
            composite_depth = torch.nan_to_num(composite_depth, float('inf'))
        composite_depth = torch.clamp(composite_depth, torch.min(depths),
                                      torch.max(depths))

        if self.white_back:
            composite_rgb = composite_rgb + 1 - weight_total

        composite_rgb = composite_rgb * 2 - 1  # Scale to (-1, 1)

        return composite_rgb, composite_depth, weights

    @torch.no_grad()
    def sample_importance(self, z_vals: torch.Tensor, weights: torch.Tensor,
                          N_importance: int) -> torch.Tensor:
        """Return depths of importance sampled points along rays.

        Args:
            z_vals (torch.Tensor): Coarse Z value (depth). Shape like
                (bz, N_points, N_depth, N_feature).
            weights (torch.Tensor): Weights of the coarse samples. Shape like
                (bz, N_points, N_depths-1, 1).
            N_importance (int): Number of samples to resample.
        """
        batch_size, num_rays, samples_per_ray, _ = z_vals.shape
        z_vals = z_vals.reshape(batch_size * num_rays, samples_per_ray)
        # -1 to account for loss of 1 sample in MipRayMarcher
        weights = weights.reshape(batch_size * num_rays, -1)

        # smooth weights as MipNeRF
        # max(weights[:-1], weights[1:])
        weights = F.max_pool1d(weights.unsqueeze(1).float(), 2, 1, padding=1)
        # 0.5 * (weights[:-1] + weights[1:])
        weights = F.avg_pool1d(weights, 2, 1).squeeze()
        weights = weights + 0.01  # add resampling padding

        z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])
        importance_z_vals = inverse_transform_sampling(z_vals_mid,
                                                       weights[:, 1:-1],
                                                       N_importance).detach()
        importance_z_vals = importance_z_vals.reshape(batch_size, num_rays,
                                                      N_importance, 1)
        return importance_z_vals


class EG3DDecoder(BaseModule):
    """Decoder for EG3D model.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels. Defaults to 32.
        hidden_channels (int): The number of channels of hidden layer.
            Defaults to 64.
        lr_multiplier (float, optional): Equalized learning rate multiplier.
            Defaults to 1.
        rgb_padding (float): Padding for RGB output. Defaults to 0.001.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int = 32,
                 hidden_channels: int = 64,
                 lr_multiplier: float = 1,
                 rgb_padding: float = 0.001):
        super().__init__()
        self.net = nn.Sequential(
            FullyConnectedLayer(
                in_channels, hidden_channels, lr_multiplier=lr_multiplier),
            nn.Softplus(),
            FullyConnectedLayer(
                hidden_channels, 1 + out_channels,
                lr_multiplier=lr_multiplier))
        self.rgb_padding = rgb_padding

    def forward(self, sampled_features: torch.Tensor) -> dict:
        """Forward function.

        Args:
            sampled_features (torch.Tensor): The sampled triplane feature for
                each points. Shape like (batch_size, xxx, xxx, n_ch).

        Returns:
            dict: A dict contains rgb feature and sigma value for each point.
        """
        sampled_features = sampled_features.mean(1)
        N, M, C = sampled_features.shape

        feat = sampled_features.view(-1, C)
        feat = self.net(feat)
        feat = feat.view(N, M, -1)

        rgb = torch.sigmoid(
            feat[..., 1:]) * (1 + 2 * self.rgb_padding) - self.rgb_padding
        sigma = feat[..., 0:1]

        return {'rgb': rgb, 'sigma': sigma}
