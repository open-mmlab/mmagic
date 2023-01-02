#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from typing import Tuple, Union

import numpy as np
import torch


class GMPICamera:

    def __init__(
        self,
        height: int = 480,
        width: int = 640,
        intrinsics: np.ndarray = None,
        ray_from_pix_center: bool = False,
    ):
        """Initialize the camera class
        Args:
            height (int, optional): [description]. Defaults to 480.
            width (int, optional): [description]. Defaults to 640.
            intrinsics (np.ndarray, optional): [description]. Defaults to None.
        """
        self._h = height
        self._w = width
        assert (
            intrinsics.ndim == 2 and intrinsics.shape[0] == 3
            and intrinsics.shape[1] == 3
        ), '[Camera] Expecting a 3x3 intrinsics matrix, but instead got {}'.format(
            intrinsics.shape)
        self._K = intrinsics

        self._ray_dir_torch_cuda = None

        self._ray_from_pix_center = ray_from_pix_center

    @property
    def intrinsic_matrix(self):
        return self._K

    @property
    def height(self):
        return self._h

    @property
    def width(self):
        return self._w

    def __repr__(self):
        return f'Camera: height={self.height}, width={self.width}, intrinsics=\n{self.intrinsic_matrix}'

    def homogeneous_coordinates(self) -> np.ndarray:
        """Construct the homogeneous coordinates [x/z, y/z, 1] for every pixel.

        Returns:
            np.ndarray: a 3 x H x W numpy ndarray corresponding to [x/z, y/z, 1]
        """
        # construct arrays of pixel coordinates
        xx, yy = np.meshgrid(
            range(int(self.width)), range(int(self.height)), indexing='xy')

        if self._ray_from_pix_center:
            # NOTE: we cast ray from pixel's center.
            xx = xx + 0.5
            yy = yy + 0.5

        # [u, v, 1] of shape [3, H, W]
        uv1 = np.stack([xx, yy, np.ones(xx.shape)])

        # [x/z, y/z, 1] of shape [3, H, W]
        inverse_K = np.linalg.inv(self.intrinsic_matrix)
        xyz_div_z = np.matmul(inverse_K, uv1.reshape(3, -1))
        xyz_div_z = xyz_div_z.reshape(3, self.height, self.width)

        return xyz_div_z

    def homogeneous_coordinates_border(self) -> np.ndarray:
        """Construct the homogeneous coordinates [x/z, y/z, 1] for every pixel.

        Returns:
            np.ndarray: a 3 x H x W numpy ndarray corresponding to [x/z, y/z, 1]
        """
        # construct arrays of pixel coordinates
        xx, yy = np.meshgrid(
            np.array([0, self.width]),
            np.array([0, self.height]),
            indexing='xy')

        # [u, v, 1] of shape [3, H, W]
        uv1 = np.stack([xx, yy, np.ones(xx.shape)])

        # [x/z, y/z, 1] of shape [3, H, W]
        inverse_K = np.linalg.inv(self.intrinsic_matrix)
        xyz_div_z = np.matmul(inverse_K, uv1.reshape(3, -1))
        xyz_div_z = xyz_div_z.reshape(3, 2, 2)

        return xyz_div_z

    def ray_dir_np(self) -> np.ndarray:
        # Construct unit-length ray directions
        xyz_div_z = self.homogeneous_coordinates
        row_l2_norms = np.linalg.norm(xyz_div_z, axis=0)
        ray_dir = xyz_div_z / row_l2_norms
        ray_dir = ray_dir.reshape(3, -1)
        return ray_dir

    def ray_dir_border_np(self) -> np.ndarray:
        # Construct unit-length ray directions
        xyz_div_z = self.homogeneous_coordinates_border
        row_l2_norms = np.linalg.norm(xyz_div_z, axis=0)
        ray_dir = xyz_div_z / row_l2_norms
        ray_dir = ray_dir.reshape(3, -1)
        return ray_dir

    def ray_dir_torch(self) -> np.ndarray:
        return torch.FloatTensor(self.ray_dir_np)

    def ray_dir_border_torch(self) -> np.ndarray:
        return torch.FloatTensor(self.ray_dir_border_np)

    def ray_dir_torch_cuda(self, device, border_only=False) -> torch.Tensor:
        if self._ray_dir_torch_cuda is None:
            if border_only:
                self._ray_dir_torch_cuda = self.ray_dir_border_torch.to(device)
            else:
                self._ray_dir_torch_cuda = self.ray_dir_torch.to(device)
        return self._ray_dir_torch_cuda

    def generate_rays(
        self,
        tf_c2w: Union[np.ndarray, torch.Tensor],
        border_only: bool = False,
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[
            np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor], ]:
        """Generate camera rays in the world space, given the camera-to-world
        transformation.

        Args:
            tf_c2w (np.ndarray): 4 x 4, camera-to-world transformation

        Returns:
            ray_dir (np.ndarray): a 3 x H x W tensor containing the unit-length ray directions for each pixel
            eye_pos (np.ndarray): a 3-vector representing the eye position
            z_dir (np.ndarray): a 3-vector representing the unit-length ray direction of the optical axis
        """
        if isinstance(tf_c2w, np.ndarray):
            return self._generate_rays_np(tf_c2w, border_only=border_only)
        elif isinstance(tf_c2w, torch.Tensor):
            return self._generate_rays_torch(tf_c2w, border_only=border_only)
        else:
            raise ValueError

    def _generate_rays_np(
        self,
        tf_c2w: np.ndarray,
        border_only: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        # extract camera-to-world transformations
        eye_pos = tf_c2w[:3, 3]
        rot_mat = tf_c2w[:3, :3]

        if border_only:
            cur_ray_dir_np = self.ray_dir_border_np
        else:
            cur_ray_dir_np = self.ray_dir_np

        # apply transformation from camera space to world space
        ray_dir = rot_mat @ cur_ray_dir_np  # 3x3 @ 3xN -> 3xN

        if border_only:
            ray_dir = ray_dir.reshape(3, 2, 2)
        else:
            ray_dir = ray_dir.reshape(3, self.height, self.width)  # 3xHxW

        # extract the z direction
        z_dir = rot_mat[:, 2]

        return ray_dir, eye_pos, z_dir

    def _generate_rays_torch(
        self,
        tf_c2w: torch.Tensor,
        border_only: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # extract camera-to-world transformations
        eye_pos = tf_c2w[:3, 3]
        rot_mat = tf_c2w[:3, :3]

        if tf_c2w.is_cuda:
            cur_ray_dir_torch = self.ray_dir_torch_cuda(
                tf_c2w.device, border_only=border_only)
        else:
            if border_only:
                cur_ray_dir_torch = self.ray_dir_border_torch
            else:
                cur_ray_dir_torch = self.ray_dir_torch

        # apply transformation from camera space to world space
        ray_dir = torch.matmul(rot_mat, cur_ray_dir_torch)  # 3x3 @ 3xN -> 3xN

        if border_only:
            ray_dir = ray_dir.reshape(3, 2, 2)  # 3xHxW
        else:
            ray_dir = ray_dir.reshape(3, self.height, self.width)  # 3xHxW

        # extract the z direction
        z_dir = rot_mat[:, 2]

        return ray_dir, eye_pos, z_dir
