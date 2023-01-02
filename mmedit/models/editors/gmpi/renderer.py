# Copyright (c) OpenMMLab. All rights reserved.
"""The renderer is a module that takes in rays, decides where to sample along
each ray, and computes pixel colors using the volume rendering equation."""

import numpy as np
import torch
from mmengine.model import BaseModule

from .cam_utils import gen_cam, gen_sphere_path
from .mpi import MPI
from .mpi_utils import (
    compute_plane_dhws_given_cam_pose_spatial_range,
    compute_plane_dhws_given_cam_pose_spatial_range_confined, sample_distance)

EPS = 1e-6


class MPIRenderer(BaseModule):

    def __init__(
            self,
            *,
            n_mpi_planes,
            plane_min_d,
            plane_max_d,
            plan_spatial_enlarge_factor,
            plane_distances_sample_method,
            cam_fov,
            sphere_center_z,
            sphere_r,
            horizontal_mean,
            horizontal_std,
            vertical_mean,
            vertical_std,
            cam_pose_n_truncated_stds,
            cam_sample_method,
            mpi_align_corners=True,
            use_xyz_ztype='depth',
            use_normalized_xyz=False,
            normalized_xyz_range='-11',
            use_confined_volume=False,
            device=torch.device('cpu'),
    ):

        self.mpi = MPI(align_corners=mpi_align_corners)

        self.use_confined_volume = use_confined_volume

        self.n_mpi_planes = n_mpi_planes
        self.plane_min_d = plane_min_d
        self.plane_max_d = plane_max_d
        self.plan_spatial_enlarge_factor = plan_spatial_enlarge_factor
        self.plane_distances_sample_method = plane_distances_sample_method

        self.mpi_tex_h = None
        self.mpi_tex_w = None

        self.cam_fov = cam_fov

        self.sphere_center = np.array([0, 0, sphere_center_z])
        self.sphere_r = sphere_r
        self.horizontal_mean = horizontal_mean
        self.horizontal_std = horizontal_std
        self.vertical_mean = vertical_mean
        self.vertical_std = vertical_std
        self.cam_pose_n_truncated_stds = cam_pose_n_truncated_stds
        self.cam_sample_method = cam_sample_method

        self.device = device

        self.compute_mpi_spatial_volume()

        self.use_xyz_ztype = use_xyz_ztype
        self.use_normalized_xyz = use_normalized_xyz
        self.normalized_xyz_range = normalized_xyz_range
        assert self.normalized_xyz_range in ['01', '-11'
                                             ], f'{self.normalized_xyz_range}'

    def set_cam(self,
                fov_deg,
                render_h,
                render_w,
                cam_ray_from_pix_center=True):
        '''
        NOTE: we need to reset camera when GAN's progressive training goes to a
        new stage, i.e., the rendering resolution changes.
        '''
        assert render_h == render_w, f'{render_h}, {render_w}'

        # We compute focal length from FOV
        # tan(fov / 2) = (w / 2) / focal ==> focal = w / (2 * tan(fov / 2))
        tan_fov_cam = np.tan(np.pi * fov_deg / (2 * 180))
        render_focal = render_w / (2 * tan_fov_cam)

        self.cam = gen_cam(
            h=render_h,
            w=render_w,
            f=render_focal,
            ray_from_pix_center=cam_ray_from_pix_center,
        )
        self.render_h = render_h
        self.render_w = render_w

    def compute_mpi_spatial_volume(self):

        plane_ds = torch.FloatTensor(
            sample_distance(
                self.plane_min_d,
                self.plane_max_d,
                self.n_mpi_planes,
                self.plane_distances_sample_method,
            ))

        # logger.info(f"\nBefore clip plane_ds: {plane_ds}\n")
        plane_ds = torch.clamp(plane_ds, self.plane_min_d, self.plane_max_d)
        # logger.info(f"\nAfter clip plane_ds: {plane_ds}\n")

        # NOTE: we define MPI in a world coordinate system with +X right +Y down +Z forward
        cam_horizontal_min = self.horizontal_mean - 1 * self.cam_pose_n_truncated_stds * self.horizontal_std
        cam_horizontal_max = self.horizontal_mean + self.cam_pose_n_truncated_stds * self.horizontal_std
        cam_vertical_min = self.vertical_mean - 1 * self.cam_pose_n_truncated_stds * self.vertical_std
        cam_vertical_max = self.vertical_mean + self.cam_pose_n_truncated_stds * self.vertical_std

        # we only need camera's ray directions,
        # so the rendering's resolution does not matter here.
        self.set_cam(self.cam_fov, 4, 4, cam_ray_from_pix_center=True)

        if self.use_confined_volume:
            compute_plane_dhws_func = compute_plane_dhws_given_cam_pose_spatial_range_confined
        else:
            compute_plane_dhws_func = compute_plane_dhws_given_cam_pose_spatial_range

        (
            plane_dhws,
            mpi_tex_expand_ratio,
        ) = compute_plane_dhws_func(
            camera=self.cam,
            sphere_center=self.sphere_center,
            sphere_r=self.sphere_r,
            cam_horizontal_min=cam_horizontal_min,
            cam_horizontal_max=cam_horizontal_max,
            cam_vertical_min=cam_vertical_min,
            cam_vertical_max=cam_vertical_max,
            cam_pose_n_truncated_stds=self.cam_pose_n_truncated_stds,
            plane_zs=plane_ds,
            enlarge_factor=self.plan_spatial_enlarge_factor,
            device=torch.device('cpu'),
        )

        self.static_mpi_plane_dhws = torch.FloatTensor(plane_dhws)
        self.dynamic_mpi_plane_dhws = self.static_mpi_plane_dhws

        # logger.info(f"static_mpi_plane_dhws: {self.static_mpi_plane_dhws}\n")

    def get_xyz(self, tex_h, tex_w, ret_single_res=True, only_z=False):
        assert tex_h == tex_w, f'Only support square resolution now. Receiving {tex_h} x {tex_w}.'
        # check they are expoenential of 2
        assert tex_h >= 4 and tex_h & (tex_w - 1) == 0, f'{tex_h}'

        if ret_single_res:
            return self.get_xyz_single_res(tex_h, tex_w, only_z=only_z)
        else:
            xyz_dict = {}
            normalized_xyz_dict = {}

            n_log2 = int(np.log2(tex_h))
            # 4, 8, ..., tex_h
            res_list = [2**i for i in range(2, n_log2 + 1)]
            for tmp_idx, tmp_res in enumerate(res_list):
                tmp_xyz, tmp_normalized_xyz = self.get_xyz_single_res(
                    tmp_res, tmp_res, only_z=only_z)
                xyz_dict[tmp_res] = tmp_xyz
                normalized_xyz_dict[tmp_res] = tmp_normalized_xyz

                if self.use_xyz_ztype == 'depth':
                    pass
                elif self.use_xyz_ztype == 'disparity':
                    xyz_dict[tmp_res][..., 2] = 1 / xyz_dict[tmp_res][..., 2]
                else:
                    raise ValueError

            return xyz_dict, normalized_xyz_dict

    def get_xyz_single_res(self, tex_h, tex_w, only_z=False):
        if only_z:
            plane_dhws = self.dynamic_mpi_plane_dhws

            # +X right, +Y down, +Z forward
            # [#planes, tex_h, tex_w, 1]
            z = plane_dhws[:, 0].reshape((-1, 1, 1, 1))

            # range [0, 1]
            normalized_z = (z - self.plane_min_d) / (
                self.plane_max_d - self.plane_min_d)
            if self.normalized_xyz_range == '-11':
                # [-1, 1]
                normalized_z = 2 * normalized_z - 1

            return z.to(self.device), normalized_z.to(self.device)
        else:
            if self.mpi_tex_h is None or self.mpi_tex_h != tex_h:
                self.comput_tex_pixels_3d_coords(tex_h, tex_w)
                self.comput_tex_pixels_3d_normalized_coords_mpi(
                    self.mpi_tex_pix_3d_coords)

            if self.use_normalized_xyz:
                return_normzlied_coords = self.mpi_tex_pix_3d_normalized_coords
            else:
                return_normzlied_coords = None

            return self.mpi_tex_pix_3d_coords[..., :3], return_normzlied_coords

    def get_xyz_interpolate_ws(self, n_src_planes, n_tgt_planes):
        raw_src_plane_ds = torch.FloatTensor(
            sample_distance(
                self.plane_min_d,
                self.plane_max_d,
                n_src_planes,
                self.plane_distances_sample_method,
            ))

        # left/right-append with placeholders
        src_plane_ds = torch.zeros(n_src_planes + 2)
        src_plane_ds[0] = -999999
        src_plane_ds[-1] = 999999
        src_plane_ds[1:-1] = raw_src_plane_ds

        tgt_plane_ds = torch.FloatTensor(
            sample_distance(
                self.plane_min_d,
                self.plane_max_d,
                n_tgt_planes,
                self.plane_distances_sample_method,
            ))

        all_ws = []
        for i in range(tgt_plane_ds.shape[0]):
            tmp_tgt_d = tgt_plane_ds[i]
            tmp_w = torch.zeros(n_src_planes + 2)
            for j in range(n_src_planes + 1):
                if src_plane_ds[j] <= tmp_tgt_d and src_plane_ds[
                        j + 1] > tmp_tgt_d:
                    tmp_range = src_plane_ds[j + 1] - src_plane_ds[j]
                    tmp_w[j] = (src_plane_ds[j + 1] - tmp_tgt_d) / (
                        tmp_range + 1e-8)
                    tmp_w[j + 1] = (tmp_tgt_d - src_plane_ds[j]) / (
                        tmp_range + 1e-8)
                    all_ws.append(tmp_w)
                    break

        # [#tgt_planes, #src_planes + 2]
        all_ws = torch.stack(all_ws, dim=0)

        return all_ws

    def comput_tex_pixels_3d_coords(self, tex_h, tex_w):

        n_planes = self.n_mpi_planes

        plane_dhws = self.dynamic_mpi_plane_dhws

        # +X right, +Y down, +Z forward
        # [#planes, tex_h, tex_w]
        z = plane_dhws[:, 0].reshape((-1, 1, 1)).expand(-1, tex_h, tex_w)

        pix_col_grid_val = torch.linspace(-1, 1, tex_w, device=z.device)
        # [#planes]
        one_side_col_len = plane_dhws[:, 2:3] / 2.0
        # [#planes, tex_w]
        tmp_x = pix_col_grid_val * one_side_col_len
        # [#planes, tex_h, tex_w]
        x = tmp_x.view((n_planes, 1, tex_w)).expand(-1, tex_h, -1)

        pix_row_grid_val = torch.linspace(-1, 1, tex_h, device=z.device)
        # [#planes]
        one_side_row_len = plane_dhws[:, 1:2] / 2.0
        # [#planes, tex_w]
        tmp_y = pix_row_grid_val * one_side_row_len
        # [#planes, tex_h, tex_w]
        y = tmp_y.view((n_planes, tex_h, 1)).expand(-1, -1, tex_w)

        # [#planes, tex_h, tex_w, 3]
        xyz = torch.stack((x, y, z), dim=-1).to(self.device)

        self.mpi_tex_h = tex_h
        self.mpi_tex_w = tex_w

        # NOTE: must compute distance here before transformation
        dist_to_concen_point = torch.norm(xyz, p=2, dim=3, keepdim=True)

        self.non_jittered_xyz = xyz.clone()

        xyz_d = torch.cat((xyz, dist_to_concen_point), dim=3)

        self.mpi_tex_pix_3d_coords = xyz_d

    def comput_tex_pixels_3d_normalized_coords_mpi(self, raw_xyz):

        # We have +X right, +Y down, +Z forward
        min_z = self.plane_min_d
        max_z = self.plane_max_d
        # x is for width
        min_x = -1 * self.static_mpi_plane_dhws[-1, 2] / 2
        max_x = self.static_mpi_plane_dhws[-1, 2] / 2
        # y is for height
        min_y = -1 * self.static_mpi_plane_dhws[-1, 1] / 2
        max_y = self.static_mpi_plane_dhws[-1, 1] / 2

        # we output xyz in [-1, 1]^3
        min_xyz = torch.FloatTensor([min_x, min_y, min_z]).reshape(
            (1, 1, 1, 3)).to(raw_xyz.device)
        max_xyz = torch.FloatTensor([max_x, max_y, max_z]).reshape(
            (1, 1, 1, 3)).to(raw_xyz.device)

        # [0, 1]
        xyz = (raw_xyz[..., :3] - min_xyz) / (max_xyz - min_xyz)

        if self.normalized_xyz_range == '-11':
            # [-1, 1]
            xyz = 2 * xyz - 1

        self.mpi_tex_pix_3d_normalized_coords = xyz

    def view_info_from_c2w_mat(self, camera, c2w, device=torch.device('cpu')):
        # [4, 4], float32
        tf_c2w = c2w
        if not isinstance(tf_c2w, torch.Tensor):
            tf_c2w = torch.FloatTensor(c2w)

        ray_dir, eye_pos, z_dir = camera.generate_rays(tf_c2w)
        # [1, 3, img_h, img_w], float32
        ray_dir = ray_dir.unsqueeze(0).float()
        # [1, 3], float32
        eye_pos = eye_pos.view(1, 3).float()
        # [1, 3], float32
        z_dir = z_dir.view(1, 3).float()
        # [1, 4, 4]
        tf_c2w = tf_c2w.unsqueeze(0)
        return ray_dir, eye_pos, z_dir, tf_c2w

    def sample_cam_poses(
        self,
        batch_size,
        horizontal_mean,
        horizontal_std,
        vertical_mean,
        vertical_std,
        random_pose,
        given_yaws=None,
        given_pitches=None,
    ):
        batch_tf_c2w, batch_yaws, batch_pitches = gen_sphere_path(
            n_cams=batch_size,
            sphere_center=self.sphere_center,
            sphere_r=self.sphere_r,
            yaw_mean=horizontal_mean,
            yaw_std=horizontal_std,
            pitch_mean=vertical_mean,
            pitch_std=vertical_std,
            n_truncated_stds=self.cam_pose_n_truncated_stds,
            flag_rnd=random_pose,
            sample_method=self.cam_sample_method,
            given_yaws=given_yaws,
            given_pitches=given_pitches,
        )

        if not isinstance(batch_tf_c2w, torch.Tensor):
            batch_tf_c2w = torch.FloatTensor(batch_tf_c2w)
        batch_tf_c2w = batch_tf_c2w.to(self.device)

        batch_ray_dir = []
        batch_eye_pos = []
        batch_z_dir = []
        for i in range(batch_tf_c2w.shape[0]):
            ray_dir, eye_pos, z_dir, tf_c2w = self.view_info_from_c2w_mat(
                self.cam, batch_tf_c2w[i, ...], device=self.device)
            batch_ray_dir.append(ray_dir)
            batch_eye_pos.append(eye_pos)
            batch_z_dir.append(z_dir)

        return (
            batch_yaws,
            batch_pitches,
            batch_tf_c2w,
            batch_ray_dir,
            batch_eye_pos,
            batch_z_dir,
        )

    def render(
        self,
        batch_mpi_rgbas,
        render_h,
        render_w,
        horizontal_mean=None,
        horizontal_std=None,
        vertical_mean=None,
        vertical_std=None,
        random_pose=True,
        given_yaws=None,
        given_pitches=None,
        given_cam_infos=None,
        assert_not_out_of_last_plane=True,
    ):

        with torch.cuda.amp.autocast(enabled=False):

            if horizontal_mean is None:
                horizontal_mean = self.horizontal_mean
            if horizontal_std is None:
                horizontal_std = self.horizontal_std
            if vertical_mean is None:
                vertical_mean = self.vertical_mean
            if vertical_std is None:
                vertical_std = self.vertical_std

            batch_size = batch_mpi_rgbas.shape[0]
            if render_h != self.render_h or render_w != self.render_w:
                self.set_cam(self.cam_fov, render_h, render_w)

            if given_cam_infos is None:
                (
                    batch_yaws,
                    batch_pitches,
                    batch_tf_c2w,
                    batch_ray_dir,
                    batch_eye_pos,
                    batch_z_dir,
                ) = self.sample_cam_poses(
                    batch_size,
                    horizontal_mean,
                    horizontal_std,
                    vertical_mean,
                    vertical_std,
                    random_pose=random_pose,
                    given_yaws=given_yaws,
                    given_pitches=given_pitches,
                )
            else:
                batch_yaws = given_cam_infos['batch_yaws']
                batch_pitches = given_cam_infos['batch_pitches']
                batch_tf_c2w = given_cam_infos['batch_tf_c2w']
                batch_ray_dir = given_cam_infos['batch_ray_dir']
                batch_eye_pos = given_cam_infos['batch_eye_pos']
                batch_z_dir = given_cam_infos['batch_z_dir']

            batch_dhws = self.dynamic_mpi_plane_dhws.reshape(
                (1, -1, 3)).expand(batch_size, -1, -1).to(self.device)
            # https://github.com/pytorch/pytorch/issues/42218
            batch_mpi_rgbas = batch_mpi_rgbas.float()
            assert (
                torch.min(batch_mpi_rgbas) >= 0.0
                and torch.max(batch_mpi_rgbas) <= 1.0
            ), f'{torch.min(batch_mpi_rgbas)}, {torch.max(batch_mpi_rgbas)}'

            batch_colors, batch_depths = self.mpi(
                batch_rgba=batch_mpi_rgbas,
                batch_dhw=batch_dhws,
                batch_ray_dir=batch_ray_dir,
                batch_eye_pos=batch_eye_pos,
                batch_z_dir=batch_z_dir,
                separate_background=None,
                assert_not_out_of_last_plane=assert_not_out_of_last_plane,
                c2w_mat=batch_tf_c2w,
                sphere_c=self.sphere_center,
            )

            # [B, 2]
            cam_anlges = torch.cat([batch_pitches, batch_yaws],
                                   -1).to(self.device)

            # [0, 1] -> [-1, 1]
            batch_colors = 2 * batch_colors - 1

        return batch_colors, batch_depths, batch_tf_c2w, cam_anlges
