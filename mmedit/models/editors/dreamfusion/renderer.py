# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn.functional as F
# TODO: move these utils to mmedit.models.utils folder
from mmedit.models.editors.eg3d.eg3d_utils import (get_ray_limits_box,
                                                   inverse_transform_sampling,
                                                   linspace_batch)
from mmedit.models.utils import normalize_vecs
from mmedit.registry import MODULES
from mmengine import print_log
from mmengine.model import BaseModule
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION

from .vanilla_nerf import NeRFNetwork


@MODULES.register_module()
class DreamFusionRenderer(BaseModule):
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
    """

    def __init__(
            self,
            # bound,
            decoder_cfg: dict,
            ray_start: float,
            ray_end: float,
            # NOTE: bound / 2, set as 2, different from eg3d
            box_warp: float = 2,
            depth_resolution: int = 64,
            depth_resolution_importance: int = 64,
            # density_noise: float = 0,  # NOTE: no use
            clamp_mode: Optional[str] = None,
            white_back: bool = True):
        super().__init__()

        self.decoder = NeRFNetwork(**decoder_cfg)

        self.ray_start = ray_start
        self.ray_end = ray_end
        self.box_warp = box_warp
        self.depth_resolution = depth_resolution
        self.depth_resolution_importance = depth_resolution_importance

        self.clamp_mode = clamp_mode
        self.white_back = white_back

        if self.white_back and self.decoder.bg_radius:
            print_log(
                'Background network in NeRF decoder will be not used '
                'since \'white_back\' is set as True.', 'current')

        # init value for shding and ambient_ratio
        self.shading = 'albedo'
        self.ambient_ratio = 1.0

    def set_shading(self, shading):
        self.shading = shading

    def set_ambient_ratio(self, ambient_ratio):
        self.ambient_ratio = ambient_ratio

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

    def forward(
        self,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        render_kwargs: Optional[dict] = dict()
    ) -> Tuple[torch.Tensor]:
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
        shading = self.get_value('shading', render_kwargs)
        ambient_ratio = self.get_value('ambient_ratio', render_kwargs)

        if ray_start == ray_end == 'auto':
            ray_start, ray_end = get_ray_limits_box(
                ray_origins, ray_directions, box_side_length=box_warp)
            is_ray_valid = ray_end > ray_start
            if torch.any(is_ray_valid).item():
                ray_start[~is_ray_valid] = ray_start[is_ray_valid].min()
                ray_end[~is_ray_valid] = ray_start[is_ray_valid].max()
        elif ray_start == ray_end == 'sphere':
            radius = ray_origins.norm(dim=-1, keepdim=True)
            ray_start = radius - self.box_warp / 2  # [B, N, 1]
            ray_end = radius + self.box_warp / 2
        else:
            assert (isinstance(ray_start, float) and isinstance(
                ray_end, float)), (
                    '\'ray_start\' and \'ray_end\' must be both float type or '
                    f'both \'auto\'. But receive {ray_start} and {ray_end}.')
            assert ray_start < ray_end, (
                '\'ray_start\' must less than \'ray_end\'.')

        # Create stratified depth samples
        depths_coarse, depth_delta = self.sample_stratified(
            ray_origins,
            ray_start,
            ray_end,
            depth_resolution,
            perturb=self.training)

        # get light direction
        if 'light_d' in render_kwargs:
            light_d = render_kwargs['light_d']
        else:
            # select random light direction
            light_d = (
                ray_origins[0, 0] +
                torch.randn(3, device=ray_start.device, dtype=torch.float))
            light_d = normalize_vecs(light_d, clamp_eps=1e-20)

        batch_size, num_rays, samples_per_ray, _ = depths_coarse.shape
        shape_prefix = [batch_size, num_rays, samples_per_ray]

        # Coarse Pass
        # [B, Res^2, N_points, 3]
        sample_coordinates = ray_origins[
            ..., None, :] + depths_coarse * ray_directions[..., None, :]

        # NOTE: add clip here
        sample_coordinates = torch.clamp(sample_coordinates,
                                         -self.box_warp / 2, self.box_warp / 2)
        out = self.neural_rendering(sample_coordinates, mode='density')
        densities_coarse = out['sigma'].reshape(*shape_prefix, -1)

        # Fine Pass
        N_importance = depth_resolution_importance
        if N_importance is not None and N_importance > 0:
            # update shape prefix
            shape_prefix_fine = (batch_size, num_rays, N_importance)
            # shape_prefix[-1] = samples_per_ray + N_importance
            with torch.no_grad():
                _, _, weights = self.volume_rendering(
                    None,
                    densities_coarse,
                    depths_coarse,
                    depth_delta,
                    mode='padding')
                depths_fine = self.sample_importance(depths_coarse, weights,
                                                     N_importance)
                # [B, Res^2, N_points, 3]
                sample_coordinates_fine = ray_origins[
                    ..., None, :] + depths_fine * ray_directions[..., None, :]

            out = self.neural_rendering(
                sample_coordinates_fine, mode='density')
            densities_fine = out['sigma']
            densities_fine = densities_fine.reshape(*shape_prefix_fine, -1)
            sort_index = self.sort_depth(depths_coarse, depths_fine)
            all_depths = self.unify_samples(depths_coarse, depths_fine,
                                            sort_index)
            all_densities = self.unify_samples(densities_coarse,
                                               densities_fine, sort_index)
            all_coord = self.unify_samples(
                sample_coordinates.reshape(*shape_prefix, -1),
                sample_coordinates_fine.reshape(*shape_prefix_fine, -1),
                sort_index)
            all_coord = torch.clamp(all_coord, -self.box_warp / 2,
                                    self.box_warp / 2)
            all_ray_directions = ray_directions[:, :,
                                                None, :].expand_as(all_coord)
            out_final = self.neural_rendering(
                all_coord, light_d, ambient_ratio, shading, mode='full')
            all_colors = out_final['color']
            all_colors = all_colors.reshape(batch_size, num_rays,
                                            samples_per_ray + N_importance, -1)
            # Aggregate
            rgb_final, depth_final, weights = self.volume_rendering(
                all_colors,
                all_densities,
                all_depths,
                depth_delta,
                mode='padding')
        else:
            # TODO: bugs here, fix later
            all_colors = out['color']
            rgb_final, depth_final, weights = self.volume_rendering(
                all_colors,
                densities_coarse,
                depths_coarse,
                depth_delta,
                mode='padding')
            out_final = out

        weights_sum = weights.sum(2)
        if self.white_back:
            bg_color = 1
        else:
            bg_color = self.decoder.forward_bg(ray_directions.reshape(-1, 3))
            bg_color = bg_color[None, ...]
        rgb_final = rgb_final + (1 - weights_sum) * bg_color

        # NOTE: we have to calculate some loss terms in renderer, maybe we
        # can find some better way to tackle this
        if self.training:
            loss_dict = dict()
            # orientation loss
            normals = out_final.get('normal', None)
            if normals is not None:
                normals = normals.view(batch_size * num_rays, -1, 3)
                # NOTE: just a dirty way to reshape
                ray_d = all_ray_directions.view(batch_size * num_rays, -1, 3)
                weights_ = weights.view(batch_size * num_rays, -1)
                loss_orient = weights_.detach() * (
                    normals * ray_d).sum(-1).clamp(min=0)**2
                loss_dict['loss_orient'] = loss_orient.sum(-1).mean()

                # surface normal smoothness
                normals_perturb = self.decoder.normal(
                    all_coord + torch.randn_like(all_coord) * 1e-2).view(
                        batch_size * num_rays, -1, 3)
                loss_smooth = (normals - normals_perturb).abs()
                loss_dict['loss_smooth'] = loss_smooth.mean()

            return rgb_final, depth_final, weights.sum(2), loss_dict

        return rgb_final, depth_final, weights.sum(2)

    def sample_stratified(self, ray_origins: torch.Tensor,
                          ray_start: Union[float, torch.Tensor],
                          ray_end: Union[float,
                                         torch.Tensor], depth_resolution: int,
                          perturb: bool) -> torch.Tensor:
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
                (bz, NeRF_res * NeRF_res, N_depth, 1). If padding is True, the
                shape will be (bz, NeRF_res * NeRF_res, N_depth+1, 1)
            ---> return padding and depths
        """
        N, M, _ = ray_origins.shape
        if isinstance(ray_start, torch.Tensor):
            # perform linspace for batch of tensor
            depths = linspace_batch(ray_start, ray_end, depth_resolution)
            depths = depths.permute(1, 2, 0, 3)
            depth_delta = (ray_end - ray_start) / (depth_resolution - 1)
            if perturb:
                # NOTE: this is different from EG3D
                # depths += torch.rand_like(depths) * depth_delta[..., None]
                depths += (torch.rand_like(depths) - 0.5) * depth_delta[...,
                                                                        None]
            return depths, depth_delta[..., None]
        else:
            depths = torch.linspace(
                ray_start,
                ray_end,
                depth_resolution,
                device=ray_origins.device)
            depths = depths.reshape(1, 1, depth_resolution, 1)
            depths = depths.repeat(N, M, 1, 1)
            depth_delta = (ray_end - ray_start) / (depth_resolution - 1)
            if perturb:
                # NOTE: this is different from EG3D
                # depths += torch.rand_like(depths) * depth_delta
                depths += (torch.rand_like(depths) - 0.5) * depth_delta[...,
                                                                        None]
            return depths, depth_delta * torch.ones(N, M, 1, 1)

    def neural_rendering(self,
                         sample_coordinates: torch.Tensor,
                         light_d: Optional[torch.Tensor] = None,
                         ratio: Optional[float] = None,
                         shading: str = 'albedo',
                         mode='density') -> dict:
        """Predict RGB features (or albedo) and densities of the coordinates by
        neural renderer model.

        Args:
            sample_coordinates (torch.Tensor): Coordinates of the sampling
                points, shape like (bz, N_depth * NeRF_res * NeRF_res, 1).
            light_d (torch.Tensor): The direction vector of light.
            ratio (float, optional): The ambident ratio in shading.
            mode (str): The forward mode of the neural renderer model.
                Supported choices are 'density' and 'full'. Defaults to
                'density'.

        Returns:
            dict: A dict contains RGB features ('rgb'), densities ('sigma')
                and normal.
        """
        xyzs = sample_coordinates.reshape(-1, 3)
        if mode == 'density':
            out = self.decoder.density(xyzs)
        else:
            out = self.decoder(
                xyzs, light_d, ambient_ratio=ratio, shading=shading)
        return out

    def unify_samples(self, target_1: torch.Tensor, target_2: torch.Tensor,
                      indices: torch.Tensor) -> torch.Tensor:
        """Unify two input tensor to one with the passed indice.

        Args:
            target_1 (torch.Tensor): The first tensor to unify.
            target_2 (torch.Tensor): The second tensor to unify.
            indices (torch.Tensor): The index of the element in the first and
                the second tensor after unifing.

        Returns:
            torch.Tensor: The unified tensor.
        """
        all_targets = torch.cat([target_1, target_2], dim=-2)
        all_targets = torch.gather(
            all_targets, -2, indices.expand(-1, -1, -1, all_targets.shape[-1]))
        return all_targets

    def sort_depth(self, depths_c: torch.Tensor,
                   depths_f: torch.Tensor) -> torch.Tensor:
        """Sort the coarse depth and fine depth, and return a indices tensor.

        Returns:
            torch.Tensor: The index of the depth in the first and the second
                tensor after unifing.
        """
        all_depths = torch.cat([depths_c, depths_f], dim=-2)

        _, indices = torch.sort(all_depths, dim=-2)
        return indices

    def volume_rendering(self,
                         colors: torch.Tensor,
                         densities: torch.Tensor,
                         depths: torch.Tensor,
                         depths_delta: Optional[torch.Tensor] = None,
                         mode='mid') -> Tuple[torch.Tensor]:
        """Volume rendering.

        Args:
            colors (torch.Tensor): Color feature for each points. Shape like
                (bz, N_points, N_depth, N_feature).
            densities (torch.Tensor): Density for each points. Shape like
                (bz, N_points, N_depth, 1).
            depths (torch.Tensor): Depths for each points. Shape like
                (bz, N_points, N_depth, 1).
            depths_delta (torch.Tensor, optional): The distance between two
                points on each ray. Shape like (bz, N_points, 1, 1)
            mode (str): The volume rendering mode. Supported choices are
                'padding' and 'mid'. If mode is 'padding', the distance
                between the last two render points will be set as
                'depths_delta'. Otherwise, will calculate the color,
                depths and density of middle points of original render points,
                and then conduct volume rendering upon the middle points.
                Defaults to 'mid'.

        Returns:
            Tuple[torch.Tensor]: A tuple of color feature
                `(bz, N_points, N_feature)`, weighted depth
                `(bz, N_points, 1)` and weight
                `(bz, N_points, N_depth-1, 1)`.
        """
        # NOTE: density and depth must not be None, colors may be None
        if mode == 'mid':
            deltas = depths[:, :, 1:] - depths[:, :, :-1]
            depths_ = (depths[:, :, :-1] + depths[:, :, 1:]) / 2
            densities_ = (densities[:, :, :-1] + densities[:, :, 1:]) / 2
        else:
            # NOTE: dreamfusion is different from EG3D ones
            assert depths_delta is not None
            # depths_ = torch.cat([depths, depths_padding], dim=2)
            deltas = depths[:, :, 1:] - depths[:, :, :-1]
            deltas = torch.cat([deltas, depths_delta], dim=2)
            depths_, densities_ = depths, densities

        # NOTE: do not use clamp for density
        if self.clamp_mode == 'softplus':
            # activation bias of -1 makes things initialize better
            densities_ = F.softplus(densities_ - 1)
        else:
            assert self.clamp_mode is None, (
                f'{self.__class__.__name__} only supports \'softplus\' for '
                f'\'clamp_mode\' but receive \'{self.clamp_mode}\'.')

        density_delta = densities_ * deltas

        alpha = 1 - torch.exp(-density_delta)

        alpha_shifted = torch.cat(
            [torch.ones_like(alpha[:, :, :1]), 1 - alpha + 1e-10], -2)
        weights = alpha * torch.cumprod(alpha_shifted, -2)[:, :, :-1]
        weight_total = weights.sum(2)

        composite_depth = torch.sum(weights * depths_, -2) / weight_total

        # clip the composite to min/max range of depths
        if digit_version(TORCH_VERSION) < digit_version('1.8.0'):
            composite_depth[torch.isnan(composite_depth)] = float('inf')
        else:
            composite_depth = torch.nan_to_num(composite_depth, float('inf'))
        composite_depth = torch.clamp(composite_depth, torch.min(depths_),
                                      torch.max(depths_))

        # NOTE: move bg to forward, since we cannot forward bg decoder in
        # volume rendering
        if colors is not None:
            if mode == 'mid':
                colors_ = (colors[:, :, :-1] + colors[:, :, 1:]) / 2
            else:
                colors_ = colors

            composite_rgb = torch.sum(weights * colors_, -2)
        else:
            composite_rgb = None

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
