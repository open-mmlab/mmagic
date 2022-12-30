# Copyright (c) OpenMMLab. All rights reserved.
import math
import random
from typing import Optional, Tuple, Union

import numpy as np
import torch
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION

from mmedit.models.utils import normalize_vecs
from mmedit.registry import MODULES

DeviceType = Optional[Union[str, int]]
VectorType = Optional[Union[list, torch.Tensor]]


def get_view_direction(thetas, phis, overhead, front):
    # NOTE: thetas and phis is inverse with ours
    #                   phis [B,];          thetas: [B,]
    # front = 0         [0, front)
    # side (left) = 1   [front, 180)
    # back = 2          [180, 180+front)
    # side (right) = 3  [180+front, 360)
    # top = 4                               [0, overhead]
    # bottom = 5                            [180-overhead, 180]
    res = torch.zeros(thetas.shape[0], dtype=torch.long)
    # first determine by phis
    res[(phis < front)] = 0
    res[(phis >= front) & (phis < np.pi)] = 1
    res[(phis >= np.pi) & (phis < (np.pi + front))] = 2
    res[(phis >= (np.pi + front))] = 3
    # override by thetas
    res[thetas <= overhead] = 4
    res[thetas >= (np.pi - overhead)] = 5
    return res


@MODULES.register_module()
class DreamFusionCamera(object):

    def __init__(self,
                 horizontal_mean,
                 vertical_mean,
                 horizontal_std,
                 vertical_std,
                 fov_mean,
                 fov_std,
                 radius_mean,
                 radius_std,
                 uniform_sphere_rate=0.5,
                 jitter_pose=False):

        self.horizontal_mean = horizontal_mean
        self.vertical_mean = vertical_mean
        self.horizontal_std = horizontal_std
        self.vertical_std = vertical_std
        self.look_at = torch.FloatTensor([0, 0, 0])
        self.up = torch.FloatTensor([0, 1, 0])
        self.radius_mean = radius_mean
        self.radius_std = radius_std
        self.fov_mean = fov_mean
        self.fov_std = fov_std

        self.uniform_sphere_rate = uniform_sphere_rate
        self.jitter_pose = jitter_pose

    def _sample_in_range(self, mean: float, std: float, batch_size: int,
                         sampling_statregy) -> torch.Tensor:
        """Sample value with specific mean and std.

        Args:
            mean (float): Mean of the sampled value.
            std (float): Standard deviation of the sampled value.
            batch_size (int): The batch size of the sampled result.

        Returns:
            torch.Tensor: Sampled results.
        """
        if sampling_statregy.upper() == 'UNIFORM':
            return (torch.rand((batch_size, 1)) - 0.5) * 2 * std + mean
        elif sampling_statregy.upper() == 'GAUSSIAN':
            return torch.randn((batch_size, 1)) * std + mean
        else:
            raise ValueError(
                'Only support \'Uniform\' sampling and \'Gaussian\' sampling '
                'currently. If you want to implement your own sampling '
                'method, you can overwrite \'_sample_in_range\' function by '
                'yourself.')

    def sample_intrinsic(
            self,
            #  fov: Optional[float] = None,
            fov_mean=None,
            fov_std=None,
            focal: Optional[float] = None,
            device: Optional[DeviceType] = None,
            batch_size: int = 1) -> torch.Tensor:
        """Sample intrinsic matrix.

        Args:
            fov (Optional[float], optional): FOV (field of view) in degree. If
                not passed, :attr:`self.fov` will be used. Defaults to None.
            focal (Optional[float], optional): Focal in pixel. If
                not passed, :attr:`self.focal` will be used. Defaults to None.
            batch_size (int): The batch size of the output. Defaults to 1.
            device (DeviceType, optional): Device to put the intrinstic
                matrix. If not passed, :attr:`self.device` will be used.
                Defaults to None.

        Returns:
            torch.Tensor: Intrinsic matrix.
        """
        fov_mean = self.fov_mean if fov_mean is None else fov_mean
        fov_std = self.fov_std if fov_std is None else fov_std
        assert not ((fov_mean is None) ^ (fov_std is None))

        if fov_mean is not None and fov_std is not None:
            fov = self._sample_in_range(fov_mean, fov_std, batch_size,
                                        'uniform')
        else:
            fov = None

        # 1. check if foc and focal is both passed
        assert (fov is None) or (focal is None), (
            '\'fov\' and focal should not be passed at the same time.')
        # 2. if fov and focal is neither not passed, use initialized ones.
        if fov is None and focal is None:
            # do not use self.fov since fov is not defined
            # fov = self.fov if fov is None else fov
            focal = self.focal if focal is None else focal

        if fov is None and focal is None:
            raise ValueError(
                '\'fov\', \'focal\', \'self.fov\' and \'self.focal\' should '
                'not be None neither.')

        if fov is not None:
            intrinstic = self.fov_to_intrinsic(fov, device)
        else:
            intrinstic = self.focal_to_instrinsic(focal, device)
        return intrinstic[None, ...].repeat(batch_size, 1, 1)

    def focal_to_instrinsic(self,
                            focal: Optional[float] = None,
                            device: DeviceType = None) -> torch.Tensor:
        """Calculate intrinsic matrix from focal.

        Args:
            focal (Optional[float], optional): Focal in degree. If
                not passed, :attr:`self.focal` will be used. Defaults to None.
            device (DeviceType, optional): Device to put the intrinsic
                matrix. If not passed, :attr:`self.device` will be used.
                Defaults to None.

        Returns:
            torch.Tensor: Intrinsic matrix.
        """
        focal = self.focal if focal is None else focal
        assert focal is not None, (
            '\'focal\' and \'self.focal\' should not be None at the '
            'same time.')
        # device = self.device if device is None else device
        # intrinsics = [[focal, 0, self.center_x], [0, focal, self.center_y],
        #               [0, 0, 1]]
        intrinsics = [[focal, 0, 0.5], [0, focal, 0.5], [0, 0, 1]]
        intrinsics = torch.tensor(intrinsics, device=device)
        return intrinsics

    def fov_to_intrinsic(self,
                         fov: Optional[float] = None,
                         device: DeviceType = None) -> torch.Tensor:
        """Calculate intrinsic matrix from FOV (field of view).

        Args:
            fov (Optional[float], optional): FOV (field of view) in degree. If
                not passed, :attr:`self.fov` will be used. Defaults to None.
            device (DeviceType, optional): Device to put the intrinstic
                matrix. If not passed, :attr:`self.device` will be used.
                Defaults to None.

        Returns:
            torch.Tensor: Intrinsic matrix.
        """
        fov = self.fov if fov is None else fov
        assert fov is not None, (
            '\'fov\' and \'self.fov\' should not be None at the same time.')
        # focal = float(self.H / (math.tan(fov * math.pi / 360)))
        focal = float(1 / (math.tan(fov * math.pi / 360)))
        intrinsics = [[focal, 0, 0.5], [0, focal, 0.5], [0, 0, 1]]
        intrinsics = torch.tensor(intrinsics, device=device)
        return intrinsics

    def sample_theta(self, mean: float, std: float,
                     batch_size: int) -> torch.Tensor:
        """Sampling the theta (yaw).

        Args:
            mean (float): Mean of theta.
            std (float): Standard deviation of theta.
            batch_size (int): Target batch size of theta.

        Returns:
            torch.Tensor: Sampled theta.
        """
        h = self._sample_in_range(mean, std, batch_size, 'uniform')
        return h

    def sample_phi(self, mean: float, std: float,
                   batch_size: int) -> torch.Tensor:
        """Sampling the phi (pitch). Unlike sampling theta, we uniformly sample
        phi on cosine space to release a spherical uniform sampling.

        Args:
            mean (float): Mean of phi.
            std (float): Standard deviation of phi.
            batch_size (int): Target batch size of phi.

        Returns:
            torch.Tensor: Sampled phi.
        """
        v = self._sample_in_range(mean, std, batch_size, 'uniform')

        # return a angular uniform sampling with `1-self.uniform_sphere_rate`
        if random.random() < (1 - self.uniform_sphere_rate):
            return v

        v = torch.clamp(v, 1e-5, math.pi - 1e-5)

        v = v / math.pi
        if digit_version(TORCH_VERSION) <= digit_version('1.6.0'):
            import numpy as np
            phi = torch.from_numpy(np.arccos((1 - 2 * v).numpy()))
        else:
            phi = torch.arccos(1 - 2 * v)
        return phi

    def sample_camera2world(self,
                            h_mean: Optional[float] = None,
                            v_mean: Optional[float] = None,
                            h_std: Optional[float] = None,
                            v_std: Optional[float] = None,
                            look_at: VectorType = None,
                            up: VectorType = None,
                            r_mean=None,
                            r_std=None,
                            batch_size: int = 1,
                            device: Optional[str] = None,
                            return_pose=False):

        # parse input
        h_mean = self.horizontal_mean if h_mean is None else h_mean
        v_mean = self.vertical_mean if v_mean is None else v_mean
        h_std = self.horizontal_std if h_std is None else h_std
        v_std = self.vertical_std if v_std is None else v_std
        r_mean = self.radius_mean if r_mean is None else r_mean
        r_std = self.radius_std if r_std is None else r_std

        look_at = self.look_at if look_at is None else look_at
        if not isinstance(look_at, torch.FloatTensor):
            look_at = torch.FloatTensor(look_at)
        look_at = look_at.to(device)
        up = self.up if up is None else up
        if not isinstance(up, torch.FloatTensor):
            up = torch.FloatTensor(up)
        up = up.to(device)

        radius = self._sample_in_range(r_mean, r_std, batch_size,
                                       'Uniform').to(device)

        theta = self.sample_theta(h_mean, h_std, batch_size).to(device)
        phi = self.sample_phi(v_mean, v_std, batch_size).to(device)
        # construct camera origin
        camera_origins = torch.zeros((batch_size, 3), device=device)

        camera_origins[:, 0:1] = radius * torch.sin(phi) * torch.cos(math.pi -
                                                                     theta)
        camera_origins[:, 2:3] = radius * torch.sin(phi) * torch.sin(math.pi -
                                                                     theta)
        camera_origins[:, 1:2] = radius * torch.cos(phi)

        # add noise jitter to camera origins, look_at and up
        if self.jitter_pose:
            camera_origins = camera_origins + (
                torch.rand_like(camera_origins) * 0.2 - 0.1)
            look_at = look_at + torch.randn_like(camera_origins) * 0.2

        # calculate forward vector and camer2world
        forward_vectors = normalize_vecs(look_at - camera_origins)
        camera2world = create_cam2world_matrix(forward_vectors, camera_origins,
                                               up)

        if return_pose:
            # NOTE: phi shape like [bz, 1], squeeze manually
            if phi.ndim == 2:
                phi = phi[:, 0]
            if theta.ndim == 2:
                theta = theta[:, 0]
            pose_index = get_view_direction(
                phi, theta, overhead=3.141 / 6, front=3.141 / 3)
            return camera2world, pose_index
        return camera2world

    def interpolation(self,
                      num_frames: int,
                      batch_size: int,
                      device='cuda') -> Tuple[list, list, list]:
        """Interpolation camera-to-world matrix in theta.

        Args:
            num_frames (int): _description_
            batch_size (int): _description_
            device (str, optional): _description_. Defaults to 'cuda'.

        Returns:
            Tuple[list, list, list]: _description_
        """
        # circle pose from sd-dreamfusion
        tmp_flag = self.uniform_sphere_rate
        self.uniform_sphere_rate = -1

        cam2world_list, pose_list, intrinsic_list = [], [], []
        # intrinsic are same, across interpolation
        intrinsic = self.sample_intrinsic(
            fov_std=0, batch_size=batch_size, device=device)
        for idx in range(num_frames):
            # NOTE: >>> follow sd-dreamfusion
            theta = (idx / num_frames) * 2 * 3.141
            phi = 3.141 / 3  # 60 degree
            radius = (self.radius_mean + self.radius_std) * 1.2
            # NOTE: <<< follow sd-dreamfusion

            cam2world, pose = self.sample_camera2world(
                h_mean=theta,
                h_std=0,
                v_mean=phi,
                v_std=0,
                r_mean=radius * 1.2,
                r_std=0,
                batch_size=batch_size,
                return_pose=True,
                device=device)

            pose_list.append(pose)
            cam2world_list.append(cam2world)
            intrinsic_list.append(intrinsic)

        self.uniform_sphere_rate = tmp_flag

        return cam2world_list, pose_list, intrinsic_list


def create_cam2world_matrix(forward_vector: torch.Tensor, origin: torch.Tensor,
                            up: torch.Tensor) -> torch.Tensor:
    """Calculate camera-to-world matrix from camera's forward vector, world
    origin and world up direction. The calculation is performed in right-hand
    coordinate system and the returned matrix is in homogeneous coordinates
    (shape like (bz, 4, 4)).

    Args:
        forward_vector (torch.Tensor): The forward vector of the camera.
        origin (torch.Tensor): The origin of the world coordinate.
        up (torch.Tensor): The up direction of the world coordinate.

    Returns:
        torch.Tensor: Camera-to-world matrix.
    """

    forward_vector = normalize_vecs(forward_vector)
    up_vector = up.type(torch.float).expand_as(forward_vector)
    right_vector = -normalize_vecs(
        torch.cross(up_vector, forward_vector, dim=-1))
    up_vector = normalize_vecs(
        torch.cross(forward_vector, right_vector, dim=-1))

    rotation_matrix = torch.eye(
        4, device=origin.device).unsqueeze(0).repeat(forward_vector.shape[0],
                                                     1, 1)
    rotation_matrix[:, :3, :3] = torch.stack(
        (right_vector, up_vector, forward_vector), axis=-1)

    translation_matrix = torch.eye(
        4, device=origin.device).unsqueeze(0).repeat(forward_vector.shape[0],
                                                     1, 1)
    translation_matrix[:, :3, 3] = origin
    cam2world = (translation_matrix @ rotation_matrix)[:, :, :]
    assert (cam2world.shape[1:] == (4, 4))
    return cam2world


def circle_poses(device,
                 radius=1.25,
                 theta=60,
                 phi=0,
                 return_dirs=False,
                 angle_overhead=30,
                 angle_front=60):
    import numpy as np
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)
    angle_overhead = np.deg2rad(angle_overhead)
    angle_front = np.deg2rad(angle_front)

    thetas = torch.FloatTensor([theta]).to(device)
    phis = torch.FloatTensor([phi]).to(device)

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ],
                          dim=-1)  # [B, 3]

    # lookat
    forward_vector = -safe_normalize(centers)
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0)
    right_vector = safe_normalize(
        torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = safe_normalize(
        torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector),
                                   dim=-1)
    poses[:, :3, 3] = centers

    if return_dirs:
        dirs = get_view_direction(thetas, phis, angle_overhead, angle_front)
    else:
        dirs = None

    return poses, dirs


# TODO: replace with ours later
def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(
        torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))
