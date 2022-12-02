# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import List, Optional, Union

import torch
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION

from mmedit.models.utils import normalize_vecs
from mmedit.registry import MODULES

DeviceType = Optional[Union[str, int]]
VectorType = Optional[Union[list, torch.Tensor]]


class BaseCamera(object):
    """Base camera class. Sample camera position on sphere with specific
    distribution (e.g., Gaussian, Uniform) and return camera-to-world matrix
    and intrinsics matrix.

    Args:
        horizontal_mean (Optional[float]): Mean of the horizontal range in
            radian. Defaults to None.
        vertical_mean (Optional[float]): Mean of the vertical range in radian.
            Defaults to None.
        horizontal_std (Optional[float]): Standard deviation of the horizontal
            range in radian. Defaults to None.
        vertical_std (Optional[float]): Standard deviation of the vertical
            range in radian. Defaults to None.
        look_at (Optional[List, torch.Tensor]): The look at position of the
            camera. Defaults to None.
        fov (Optional[float]): The FOV (field-of-view) in degree. Defaults
            to None.
        up (Optional[List, torch.Tensor]): The up direction of the world
            coordinate. Defaults to None.
        radius (Optional[float]): Radius of the sphere. Defaults to None.
        sampling_strategy (Optional[str]): The sampling strategy (distribution)
            of the camera. Support 'Uniform' and 'Gaussian'.
            Defaults to 'Uniform'.
    """

    def __init__(self,
                 horizontal_mean: Optional[float] = None,
                 vertical_mean: Optional[float] = None,
                 horizontal_std: Optional[float] = 0,
                 vertical_std: Optional[float] = 0,
                 look_at: VectorType = [0, 0, 0],
                 fov: Optional[float] = None,
                 focal: Optional[float] = None,
                 up: VectorType = [0, 1, 0],
                 radius: Optional[float] = 1,
                 sampling_strategy: str = 'uniform'):
        super().__init__()
        self.horizontal_mean = horizontal_mean
        self.vertical_mean = vertical_mean
        self.horizontal_std = horizontal_std
        self.vertical_std = vertical_std
        self.look_at = look_at
        self.up = up
        self.radius = radius
        self.sampling_statregy = sampling_strategy

        assert ((fov is None) or (focal is None)), (
            '\'fov\' and \'focal\' should not be passed at the same time.')
        self.fov = fov
        self.focal = focal

    def _sample_in_range(self, mean: float, std: float,
                         batch_size: int) -> torch.Tensor:
        """Sample value with specific mean and std.

        Args:
            mean (float): Mean of the sampled value.
            std (float): Standard deviation of the sampled value.
            batch_size (int): The batch size of the sampled result.

        Returns:
            torch.Tensor: Sampled results.
        """
        if self.sampling_statregy.upper() == 'UNIFORM':
            return (torch.rand((batch_size, 1)) - 0.5) * 2 * std + mean
        elif self.sampling_statregy.upper() == 'GAUSSIAN':
            return torch.randn((batch_size, 1)) * std + mean
        else:
            raise ValueError(
                'Only support \'Uniform\' sampling and \'Gaussian\' sampling '
                'currently. If you want to implement your own sampling '
                'method, you can overwrite \'_sample_in_range\' function by '
                'yourself.')

    def sample_intrinsic(self,
                         fov: Optional[float] = None,
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
        # 1. check if foc and focal is both passed
        assert (fov is None) or (focal is None), (
            '\'fov\' and focal should not be passed at the same time.')
        # 2. if fov and focal is neither not passed, use initialized ones.
        if fov is None and focal is None:
            fov = self.fov if fov is None else fov
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
        # device = self.device if device is None else device
        # NOTE: EG3D multpile '1 / 1.414' as `image_width` to `focal`, we
        # retain this operation
        focal = float(1 / (math.tan(fov * math.pi / 360) * 1.414))
        intrinsics = [[focal, 0, 0.5], [0, focal, 0.5], [0, 0, 1]]
        intrinsics = torch.tensor(intrinsics, device=device)
        return intrinsics

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
        h = self._sample_in_range(mean, std, batch_size)
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
        v = self._sample_in_range(mean, std, batch_size)
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
                            radius: Optional[float] = None,
                            batch_size: int = 1,
                            device: Optional[str] = None) -> torch.Tensor:
        """Sample camera-to-world matrix with the passed condition.

        Args:
            h_mean (Optional[float], optional): Mean of horizontal range in
                radian. Defaults to None.
            v_mean (Optional[float], optional): Mean of vertical range in
                radian. Defaults to None.
            h_std (Optional[float], optional): Standard deviation of
                horizontal in radian. Defaults to None.
            v_std (Optional[float], optional): Standard deviation of
                horizontal in radian. Defaults to None.
            look_at (Optional[Tuple[list, torch.Tensor]], optional): Look-at
                position. Defaults to None.
            up (Optional[Tuple[list, torch.Tensor]], optional): Up direction
                of the world coordinate. Defaults to None.
            radius (Optional[float]): Radius of the sphere. Defaults to None.
            batch_size (int, optional): Batch size of the results.
                Defaults to 1.
            device (Optional[str], optional): The target device of the results.
                Defaults to None.

        Returns:
            torch.Tensor: Sampled camera-to-world matrix.
        """
        # parse input
        h_mean = self.horizontal_mean if h_mean is None else h_mean
        v_mean = self.vertical_mean if v_mean is None else v_mean
        h_std = self.horizontal_std if h_std is None else h_std
        v_std = self.vertical_std if v_std is None else v_std
        radius = self.radius if radius is None else radius
        # device = self.device if device is None else device
        look_at = self.look_at if look_at is None else look_at
        if not isinstance(look_at, torch.FloatTensor):
            look_at = torch.FloatTensor(look_at)
        look_at = look_at.to(device)
        up = self.up if up is None else up
        if not isinstance(up, torch.FloatTensor):
            up = torch.FloatTensor(up)
        up = up.to(device)

        # sample yaw and pitch
        theta = self.sample_theta(h_mean, h_std, batch_size).to(device)
        phi = self.sample_phi(v_mean, v_std, batch_size).to(device)
        # construct camera origin
        camera_origins = torch.zeros((batch_size, 3), device=device)

        camera_origins[:, 0:1] = radius * torch.sin(phi) * torch.cos(math.pi -
                                                                     theta)
        camera_origins[:, 2:3] = radius * torch.sin(phi) * torch.sin(math.pi -
                                                                     theta)
        camera_origins[:, 1:2] = radius * torch.cos(phi)
        # calculate forward vector and camer2world
        forward_vectors = normalize_vecs(look_at - camera_origins)
        camera2world = create_cam2world_matrix(forward_vectors, camera_origins,
                                               up)
        return camera2world

    def interpolation_cam2world(self,
                                num_images: int,
                                h_mean: Optional[float] = None,
                                v_mean: Optional[float] = None,
                                h_std: Optional[float] = None,
                                v_std: Optional[float] = None,
                                look_at: VectorType = None,
                                up: VectorType = None,
                                radius: Optional[float] = None,
                                batch_size: int = 1,
                                device: Optional[str] = None
                                ) -> List[torch.Tensor]:
        """Interpolation camera original in spherical trajectory and return a
        list of camera-to-world matrix.

        Args:
            num_images (int): The number of images in interpolation.
            h_mean (Optional[float], optional): Mean of horizontal range in
                radian. Defaults to None.
            v_mean (Optional[float], optional): Mean of vertical range in
                radian. Defaults to None.
            h_std (Optional[float], optional): Standard deviation of
                horizontal in radian. Defaults to None.
            v_std (Optional[float], optional): Standard deviation of
                horizontal in radian. Defaults to None.
            look_at (Optional[Tuple[list, torch.Tensor]], optional): Look-at
                position. Defaults to None.
            up (Optional[Tuple[list, torch.Tensor]], optional): Up direction
                of the world coordinate. Defaults to None.
            radius (Optional[float]): Radius of the sphere. Defaults to None.
            batch_size (int, optional): Batch size of the results.
                Defaults to 1.
            device (Optional[str], optional): The target device of the results.
                Defaults to None.

        Returns:
            List[torch.Tensor]: List of sampled camera-to-world matrix.
        """
        h_mean = self.horizontal_mean if h_mean is None else h_mean
        v_mean = self.vertical_mean if v_mean is None else v_mean
        h_std = self.horizontal_std if h_std is None else h_std
        v_std = self.vertical_std if v_std is None else v_std
        radius = self.radius if radius is None else radius
        look_at = self.look_at if look_at is None else look_at
        if not isinstance(look_at, torch.FloatTensor):
            look_at = torch.FloatTensor(look_at)
        look_at = look_at.to(device)
        up = self.up if up is None else up
        if not isinstance(up, torch.FloatTensor):
            up = torch.FloatTensor(up)
        up = up.to(device)

        cam2world_list = []
        for idx in range(num_images):
            h = h_mean + h_std * math.sin(2 * math.pi / num_images * idx)
            v = v_mean + v_std * math.cos(2 * math.pi / num_images * idx)
            cam2world = self.sample_camera2world(
                h_mean=h,
                v_mean=v,
                h_std=0,
                v_std=0,
                batch_size=batch_size,
                device=device)
            cam2world_list.append(cam2world)

        return cam2world_list

    def __repr__(self):
        repr_string = f'{self.__class__.__name__}'
        attribute_list = [
            'horizontal_mean', 'vertical_mean', 'horizontal_std',
            'vertical_std', 'FOV', 'focal', 'look_at', 'up', 'radius',
            'sampling_statregy'
        ]
        for attribute in attribute_list:
            if getattr(self, attribute, None) is not None:
                repr_string += f'\n    {attribute}: {getattr(self, attribute)}'
        return repr_string


@MODULES.register_module()
class GaussianCamera(BaseCamera):
    """Pre-defined camera class. Sample camera position in gaussian
    distribution.

    Args:
        horizontal_mean (Optional[float]): Mean of the horizontal range in
            radian. Defaults to None.
        vertical_mean (Optional[float]): Mean of the vertical range in radian.
            Defaults to None.
        horizontal_std (Optional[float]): Standard deviation of the horizontal
            range in radian. Defaults to None.
        vertical_std (Optional[float]): Standard deviation of the vertical
            range in radian. Defaults to None.
        look_at (Optional[List, torch.Tensor]): The look at position of the
            camera. Defaults to None.
        up (Optional[List, torch.Tensor]): The up direction of the world
            coordinate. Defaults to None.
        radius (Optional[float]): Radius of the sphere. Defaults to None.
    """

    def __init__(self,
                 horizontal_mean: Optional[float] = None,
                 vertical_mean: Optional[float] = None,
                 horizontal_std: Optional[float] = 0,
                 vertical_std: Optional[float] = 0,
                 look_at: List = [0, 0, 0],
                 fov: Optional[float] = None,
                 focal: Optional[float] = None,
                 up: VectorType = [0, 1, 0],
                 radius: Optional[float] = 1):
        super().__init__(horizontal_mean, vertical_mean, horizontal_std,
                         vertical_std, look_at, fov, focal, up, radius,
                         'gaussian')


@MODULES.register_module()
class UniformCamera(BaseCamera):
    """Pre-defined camera class. Sample camera position in uniform
    distribution.

    Args:
        horizontal_mean (Optional[float]): Mean of the horizontal range in
            radian. Defaults to None.
        vertical_mean (Optional[float]): Mean of the vertical range in radian.
            Defaults to None.
        horizontal_std (Optional[float]): Standard deviation of the horizontal
            range in radian. Defaults to None.
        vertical_std (Optional[float]): Standard deviation of the vertical
            range in radian. Defaults to None.
        look_at (Optional[List, torch.Tensor]): The look at position of the
            camera. Defaults to None.
        up (Optional[List, torch.Tensor]): The up direction of the world
            coordinate. Defaults to None.
        radius (Optional[float]): Radius of the sphere. Defaults to None.
    """

    def __init__(self,
                 horizontal_mean: Optional[float] = None,
                 vertical_mean: Optional[float] = None,
                 horizontal_std: Optional[float] = 0,
                 vertical_std: Optional[float] = 0,
                 look_at: List = [0, 0, 0],
                 fov: Optional[float] = None,
                 focal: Optional[float] = None,
                 up: VectorType = [0, 1, 0],
                 radius: Optional[float] = 1):
        super().__init__(horizontal_mean, vertical_mean, horizontal_std,
                         vertical_std, look_at, fov, focal, up, radius,
                         'uniform')


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
