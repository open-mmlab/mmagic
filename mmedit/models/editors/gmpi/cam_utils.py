#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from typing import List, Union

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from mmedit.models.utils import normalize_vecs, truncated_normal
from .camera import Camera


def gen_cam(*, h, w, f, ray_from_pix_center):
    # Create a camera for rendering purposes
    # f = FOCAL
    # K = np.array([[f, 0.0, (w - 1) / 2], [0.0, f, (h - 1) / 2], [0.0, 0.0, 1.0]])
    K = np.array([[f, 0.0, w / 2], [0.0, f, h / 2], [0.0, 0.0, 1.0]])
    camera = Camera(
        height=h,
        width=w,
        intrinsics=K,
        ray_from_pix_center=ray_from_pix_center)
    return camera


def gen_c2w_mat(cam_pos=None, rot_mat=None):
    # NOTE: this is a dummy camera used for specifying the z-coordinate
    # Assuming positive +X (right), +Y (downward), and +Z (forward)
    if cam_pos is None:
        # cam_pos = np.array([-0.1, 0, -5])
        cam_pos = np.array([0, 0, 0])
    if rot_mat is None:
        rot_mat = np.eye(3)
    tvec = np.expand_dims(cam_pos, axis=1)
    tf_c2w = np.hstack((rot_mat, tvec))
    tf_c2w = np.vstack((tf_c2w, np.array([0, 0, 0, 1])))
    return tf_c2w


def get_start_xy_given_dir_and_end_xyz(ray_dir: np.ndarray,
                                       end_xyz: np.ndarray,
                                       start_z: float) -> np.ndarray:
    """Computes the vector's start point's XY coordinates.

    Essentially, end_point - start_point = ray_dir * | end_z - start_z |
    ==> start_point = end_point - ray_dir * | end_z - start_z |

    Args:
        ray_dir (np.ndarray): a [3, ] vector for vector's unit-length directions
        end_xyz (np.ndarray): a [3, ] vector for end point's coordinates
        start_z (float): start point's z coordinate

    Returns:
        end_xyz (np.ndarray): end point's coordinates
    """
    scale = np.abs(end_xyz[2] - start_z) / ray_dir[2]
    start_xyz = end_xyz - ray_dir * scale
    # print("end_xyz: ", end_xyz.shape, ray_dir.shape, end_xyz, start_z, scale)
    return start_xyz


def get_end_xy_given_dir_and_start_xyz(ray_dir: np.ndarray,
                                       start_xyz: np.ndarray,
                                       end_z: float) -> np.ndarray:
    """Computes the vector's end point's XY coordinates.

    Essentially, end_point - start_point = ray_dir * | end_z - start_z |
    ==> end_point = start_point + ray_dir * | end_z - start_z |

    Args:
        ray_dir (np.ndarray): a [3, ] vector for vector's unit-length directions
        end_xyz (np.ndarray): a [3, ] vector for end point's coordinates
        start_z (float): start point's z coordinate

    Returns:
        end_xyz (np.ndarray): end point's coordinates
    """
    scale = np.abs(start_xyz[2] - end_z) / ray_dir[2]
    end_xyz = start_xyz + ray_dir * scale
    # print("end_xyz: ", end_xyz.shape, ray_dir.shape, end_xyz, start_z, scale)
    return end_xyz


def compute_max_xy_range_given_cam_orientation_and_z(cam_img_rays: np.ndarray,
                                                     plane_xyz: np.ndarray,
                                                     pos_z: float
                                                     ) -> List[List[float]]:
    """This function computes the maximum possible camera's position range on XY-plane given camera's orientation and position Z.
    Essentially, the range of camera's XY coordinates are constrained as following
    (format: pixel on image plane --> MPI plane's point):
    - bottom-left --> top-right
    - top-left --> bottom-right
    - top-right --> bottom-left
    - bottom-right --> top-left

    All computations are conducted in world coordinate systems.

    Args:
        cam_img_rays: 3 x H x W, rays for every pixel in camera's image plane
        plane_xyz: the MPI's plane which must be visible for the camera
        pos_z: z coordinate for the camera's position

    Returns:
        Tuple[np.ndarray, np.ndarray]: range for x-coordiante and y-coordiante respectively.
    """

    # [3, H, W]
    assert (cam_img_rays.ndim == 3) and (cam_img_rays.shape[0] == 3)
    _, img_h, img_w = cam_img_rays.shape

    # [3, H, W]
    assert (plane_xyz.ndim == 3) and (plane_xyz.shape[0] == 3)
    _, mpi_h, mpi_w = plane_xyz.shape

    cam_img_bottom_left_xyz = get_start_xy_given_dir_and_end_xyz(
        cam_img_rays[:, img_h - 1, 0], plane_xyz[:, 0, mpi_w - 1], pos_z)
    cam_img_top_left_xyz = get_start_xy_given_dir_and_end_xyz(
        cam_img_rays[:, 0, 0], plane_xyz[:, mpi_h - 1, mpi_w - 1], pos_z)
    cam_img_top_right_xyz = get_start_xy_given_dir_and_end_xyz(
        cam_img_rays[:, 0, img_w - 1], plane_xyz[:, mpi_h - 1, 0], pos_z)
    cam_img_bottom_right_xyz = get_start_xy_given_dir_and_end_xyz(
        cam_img_rays[:, img_h - 1, img_w - 1], plane_xyz[:, 0, 0], pos_z)

    # left-most pixel of camera image defines right-most valid position
    # meanwhile, since +X points right, left-most position has smallest value
    assert (cam_img_bottom_left_xyz[0] >= cam_img_bottom_right_xyz[0]
            ), f'{cam_img_bottom_left_xyz[0]}, {cam_img_bottom_right_xyz[0]}'
    assert cam_img_top_left_xyz[0] >= cam_img_top_right_xyz[
        0], f'{cam_img_top_left_xyz[0]}, {cam_img_top_right_xyz[0]}'
    # top-most pixel of camera image defines top-most valid position
    # meanwhile, since +Y points downward, top-most position has smallest value
    assert (cam_img_bottom_left_xyz[1] <= cam_img_top_left_xyz[1]
            ), f'{cam_img_bottom_left_xyz[1]}, {cam_img_top_left_xyz[1]}'
    assert (cam_img_bottom_right_xyz[1] <= cam_img_top_right_xyz[1]
            ), f'{cam_img_bottom_right_xyz[1]}, {cam_img_top_right_xyz[1]}'

    cam_pos_bound_xyz = np.array([
        cam_img_bottom_left_xyz,
        cam_img_top_left_xyz,
        cam_img_top_right_xyz,
        cam_img_bottom_right_xyz,
    ])

    # NOTE: img_xyz_range formulates a rectangle for valid camera positions.
    # [min_x, min_y, z] is top-left corner, [max_x, max_y, z] is the bottom-right corner
    cam_pos_xyz_range_list = list(
        zip(
            np.min(cam_pos_bound_xyz, axis=0).tolist(),
            np.max(cam_pos_bound_xyz, axis=0).tolist(),
        ))
    cam_pos_xyz_range = {
        'min_x': cam_pos_xyz_range_list[0][0],
        'max_x': cam_pos_xyz_range_list[0][1],
        'min_y': cam_pos_xyz_range_list[1][0],
        'max_y': cam_pos_xyz_range_list[1][1],
        'min_z': cam_pos_xyz_range_list[2][0],
        'max_z': cam_pos_xyz_range_list[2][1],
    }

    return cam_pos_xyz_range


def compute_min_xy_range_given_cam_orientation_and_z(cam_img_rays: np.ndarray,
                                                     plane_xyz: np.ndarray,
                                                     pos_z: float
                                                     ) -> List[List[float]]:
    """This function computes the minimum possible camera's position range on XY-plane given camera's orientation and position Z.
    Essentially, the range of camera's XY coordinates are constrained as following
    (format: pixel on image plane --> MPI plane's point):
    - bottom-left --> bottom-left
    - top-left --> top-left
    - top-right --> top-right
    - bottom-right --> bottom-right

    All computations are conducted in world coordinate systems.

    Args:
        cam_img_rays: 3 x H x W, rays for every pixel in camera's image plane
        plane_xyz: the MPI's plane which must be visible for the camera
        pos_z: z coordinate for the camera's position

    Returns:
        Tuple[np.ndarray, np.ndarray]: range for x-coordiante and y-coordiante respectively.
    """

    # [3, H, W]
    assert (cam_img_rays.ndim == 3) and (cam_img_rays.shape[0] == 3)
    _, img_h, img_w = cam_img_rays.shape

    # [3, H, W]
    assert (plane_xyz.ndim == 3) and (plane_xyz.shape[0] == 3)
    _, mpi_h, mpi_w = plane_xyz.shape

    cam_img_bottom_left_xyz = get_start_xy_given_dir_and_end_xyz(
        cam_img_rays[:, img_h - 1, 0], plane_xyz[:, mpi_h - 1, 0], pos_z)
    cam_img_top_left_xyz = get_start_xy_given_dir_and_end_xyz(
        cam_img_rays[:, 0, 0], plane_xyz[:, 0, 0], pos_z)
    cam_img_top_right_xyz = get_start_xy_given_dir_and_end_xyz(
        cam_img_rays[:, 0, img_w - 1], plane_xyz[:, 0, mpi_w - 1], pos_z)
    cam_img_bottom_right_xyz = get_start_xy_given_dir_and_end_xyz(
        cam_img_rays[:, img_h - 1, img_w - 1], plane_xyz[:, mpi_h - 1,
                                                         mpi_w - 1], pos_z)

    # left-most pixel of camera image defines right-most valid position
    # meanwhile, since +X points right, left-most position has smallest value
    assert (cam_img_bottom_left_xyz[0] >= cam_img_bottom_right_xyz[0]
            ), f'{cam_img_bottom_left_xyz[0]}, {cam_img_bottom_right_xyz[0]}'
    assert cam_img_top_left_xyz[0] >= cam_img_top_right_xyz[
        0], f'{cam_img_top_left_xyz[0]}, {cam_img_top_right_xyz[0]}'
    # top-most pixel of camera image defines bottom-most valid position
    # meanwhile, since +Y points downward, top-most position has smallest value
    assert (cam_img_bottom_left_xyz[1] <= cam_img_top_left_xyz[1]
            ), f'{cam_img_bottom_left_xyz[1]}, {cam_img_top_left_xyz[1]}'
    assert (cam_img_bottom_right_xyz[1] <= cam_img_top_right_xyz[1]
            ), f'{cam_img_bottom_right_xyz[1]}, {cam_img_top_right_xyz[1]}'

    cam_pos_bound_xyz = np.array([
        cam_img_bottom_left_xyz,
        cam_img_top_left_xyz,
        cam_img_top_right_xyz,
        cam_img_bottom_right_xyz,
    ])

    # NOTE: img_xyz_range formulates a rectangle for valid camera positions.
    # [min_x, min_y, z] is top-left corner, [max_x, max_y, z] is the bottom-right corner
    cam_pos_xyz_range_list = list(
        zip(
            np.min(cam_pos_bound_xyz, axis=0).tolist(),
            np.max(cam_pos_bound_xyz, axis=0).tolist(),
        ))
    cam_pos_xyz_range = {
        'min_x': cam_pos_xyz_range_list[0][0],
        'max_x': cam_pos_xyz_range_list[0][1],
        'min_y': cam_pos_xyz_range_list[1][0],
        'max_y': cam_pos_xyz_range_list[1][1],
        'min_z': cam_pos_xyz_range_list[2][0],
        'max_z': cam_pos_xyz_range_list[2][1],
    }

    return cam_pos_xyz_range


def compute_min_visible_range_given_cam_orientation_and_z(
        cam_img_rays: np.ndarray, plane_xyz: np.ndarray,
        pos_z: float) -> List[List[float]]:
    """This function computes the minimum possible visible position range on XY-plane at position Z.
    Essentially, the range of visible XY coordinates are constrained as following
    (format: pixel on image plane --> MPI plane's point):
    - bottom-left --> bottom-left
    - top-left --> top-left
    - top-right --> top-right
    - bottom-right --> bottom-right

    All computations are conducted in world coordinate systems.

    Args:
        cam_img_rays: 3 x H x W, rays for every pixel in camera's image plane
        plane_xyz: the MPI's plane which must be visible for the camera
        pos_z: z coordinate for the camera's position

    Returns:
        Tuple[np.ndarray, np.ndarray]: range for x-coordiante and y-coordiante respectively.
    """

    # [3, H, W]
    assert (cam_img_rays.ndim == 3) and (cam_img_rays.shape[0] == 3)
    _, img_h, img_w = cam_img_rays.shape

    # [3, H, W]
    assert (plane_xyz.ndim == 3) and (plane_xyz.shape[0] == 3)
    _, mpi_h, mpi_w = plane_xyz.shape

    bottom_left_xyz = get_end_xy_given_dir_and_start_xyz(
        cam_img_rays[:, img_h - 1, 0], plane_xyz[:, mpi_h - 1, 0], pos_z)
    top_left_xyz = get_end_xy_given_dir_and_start_xyz(cam_img_rays[:, 0, 0],
                                                      plane_xyz[:, 0,
                                                                0], pos_z)
    top_right_xyz = get_end_xy_given_dir_and_start_xyz(
        cam_img_rays[:, 0, img_w - 1], plane_xyz[:, 0, mpi_w - 1], pos_z)
    bottom_right_xyz = get_end_xy_given_dir_and_start_xyz(
        cam_img_rays[:, img_h - 1, img_w - 1], plane_xyz[:, mpi_h - 1,
                                                         mpi_w - 1], pos_z)

    # left-most pixel of camera image defines left-most visible position
    # meanwhile, since +X points right, left-most position has smallest value
    assert bottom_left_xyz[0] <= bottom_right_xyz[
        0], f'{bottom_left_xyz[0]}, {bottom_right_xyz[0]}'
    assert top_left_xyz[0] <= top_right_xyz[
        0], f'{top_left_xyz[0]}, {top_right_xyz[0]}'
    # top-most pixel of camera image defines top-most valid position
    # meanwhile, since +Y points downward, top-most position has smallest value
    assert bottom_left_xyz[1] >= top_left_xyz[
        1], f'{bottom_left_xyz[1]}, {top_left_xyz[1]}'
    assert bottom_right_xyz[1] >= top_right_xyz[
        1], f'{bottom_right_xyz[1]}, {top_right_xyz[1]}'

    vis_pos_bound_xyz = np.array([
        bottom_left_xyz,
        top_left_xyz,
        top_right_xyz,
        bottom_right_xyz,
    ])

    # NOTE: img_xyz_range formulates a rectangle for valid camera positions.
    # [min_x, min_y, z] is top-left corner, [max_x, max_y, z] is the bottom-right corner
    vis_pos_xyz_range_list = list(
        zip(
            np.min(vis_pos_bound_xyz, axis=0).tolist(),
            np.max(vis_pos_bound_xyz, axis=0).tolist(),
        ))
    vis_pos_xyz_range = {
        'min_x': vis_pos_xyz_range_list[0][0],
        'max_x': vis_pos_xyz_range_list[0][1],
        'min_y': vis_pos_xyz_range_list[1][0],
        'max_y': vis_pos_xyz_range_list[1][1],
        'min_z': vis_pos_xyz_range_list[2][0],
        'max_z': vis_pos_xyz_range_list[2][1],
    }

    return vis_pos_xyz_range


def gen_zig_zag_path(
    cam_pos_xyz_range: List[List[float]],
    n_turns: int,
    n_cam_per_turn: int,
    rnd_pos: bool = True,
    single_side_invalid_area_ratio: float = 0.3,
) -> List[np.ndarray]:
    """Generate zig-zag path for camera position.

    Args:
        cam_pos_xyz_range (List[List[float]]): camera position's xyz range, within which MPI's 1st plane is visible.
            Z coordinates are the same.
        n_turns (int): #turn_arounds for this path
        n_cam_per_turn (int): #camera_pos for each turn
        rnd_pos (bool): whether to randomly sample positions
        single_side_invalid_area_ratio (float): this value constrains the area where a valid path can be sampled.
            Specifically, if this value is 0.3, then only the normalized grid from 0.3 to 0.7 can be used.

    Returns:
        List(np.ndarray): a list of camera positions along the path
    """
    min_x, max_x = cam_pos_xyz_range['min_x'], cam_pos_xyz_range['max_x']
    min_y, max_y = cam_pos_xyz_range['min_y'], cam_pos_xyz_range['max_y']
    pos_z = cam_pos_xyz_range['min_z']

    _eps_x = (max_x - min_x) * single_side_invalid_area_ratio
    _eps_y = (max_y - min_y) * single_side_invalid_area_ratio

    # narrow down the bound a little bit
    min_x = min_x + _eps_x
    max_x = max_x - _eps_x
    min_y = min_y + _eps_y
    max_y = max_y - _eps_y

    n_cam_poses = n_turns * n_cam_per_turn
    cam_pos = []

    # sample Y-coordinates:
    if rnd_pos:
        y_coords = np.sort(np.random.uniform(min_y, max_y, size=n_cam_poses))
    else:
        y_coords = np.linspace(min_y, max_y, num=n_cam_poses)

    for turn_i in np.arange(n_turns):
        if rnd_pos:
            x_coords = np.sort(
                np.random.uniform(min_x, max_x, size=n_cam_per_turn))
        else:
            x_coords = np.linspace(min_x, max_x, num=n_cam_per_turn)

        if turn_i % 2 == 0:
            x_coords = x_coords[::-1]

        for i, tmp_x_coord in enumerate(x_coords):
            pos_i = turn_i * n_cam_per_turn + i
            cam_pos.append(np.array([tmp_x_coord, y_coords[pos_i], pos_z]))

    return cam_pos


def gen_same_z_horizontal_path(
    *,
    cam_pos_xyz_range: List[List[float]],
    n_cam_poses: int,
    rnd_pos: bool = True,
    single_side_invalid_area_ratio: float = 0.3,
) -> List[np.ndarray]:
    """Generate horizontal path for camera position.

    Args:
        cam_pos_xyz_range (List[List[float]]): camera position's xyz range, within which MPI's 1st plane is visible.
            Z coordinates are the same.
        n_cam_poses (int): the number of camera poses
        rnd_pos (bool): whether to randomly sample positions
        single_side_invalid_area_ratio (float): this value constrains the area where a valid path can be sampled.
            Specifically, if this value is 0.3, then only the normalized grid from 0.3 to 0.7 can be used.

    Returns:
        List(np.ndarray): a list of camera positions along the path
    """
    min_x, max_x = cam_pos_xyz_range['min_x'], cam_pos_xyz_range['max_x']
    min_y, max_y = cam_pos_xyz_range['min_y'], cam_pos_xyz_range['max_y']
    pos_z = cam_pos_xyz_range['min_z']

    _eps_x = (max_x - min_x) * single_side_invalid_area_ratio
    _eps_y = (max_y - min_y) * single_side_invalid_area_ratio

    # narrow down the bound a little bit
    min_x = min_x + _eps_x
    max_x = max_x - _eps_x
    min_y = min_y + _eps_y
    max_y = max_y - _eps_y

    cam_pos = []

    # sample Y-coordinates:
    if rnd_pos:
        y_coords = np.random.uniform(min_y, max_y)
    else:
        y_coords = (min_y + max_y) / 2

    if rnd_pos:
        x_coords = np.sort(np.random.uniform(min_x, max_x, size=n_cam_poses))
    else:
        x_coords = np.linspace(min_x, max_x, num=n_cam_poses)

    for i, tmp_x_coord in enumerate(x_coords):
        cam_pos.append(np.array([tmp_x_coord, y_coords, pos_z]))

    return cam_pos


def gen_same_z_rnd_path(
    cam_pos_xyz_range: List[List[float]],
    n_cam_poses: int,
    single_side_invalid_area_ratio: float = 0.3,
) -> List[np.ndarray]:
    """Generate random path for camera position.

    Args:
        cam_pos_xyz_range (List[List[float]]): camera position's xyz range, within which MPI's 1st plane is visible.
            Z coordinates are the same.
        n_cam_poses (int): the number of camera poses
        rnd_pos (bool): whether to randomly sample positions
        single_side_invalid_area_ratio (float): this value constrains the area where a valid path can be sampled.
            Specifically, if this value is 0.3, then only the normalized grid from 0.3 to 0.7 can be used.

    Returns:
        List(np.ndarray): a list of camera positions along the path
    """
    min_x, max_x = cam_pos_xyz_range['min_x'], cam_pos_xyz_range['max_x']
    min_y, max_y = cam_pos_xyz_range['min_y'], cam_pos_xyz_range['max_y']
    pos_z = cam_pos_xyz_range['min_z']

    _eps_x = (max_x - min_x) * single_side_invalid_area_ratio
    _eps_y = (max_y - min_y) * single_side_invalid_area_ratio

    # narrow down the bound a little bit
    min_x = min_x + _eps_x
    max_x = max_x - _eps_x
    min_y = min_y + _eps_y
    max_y = max_y - _eps_y

    cam_pos = []

    x_coords = np.random.uniform(
        min_x, max_x, size=n_cam_poses).reshape((-1, 1))
    y_coords = np.random.uniform(
        min_y, max_y, size=n_cam_poses).reshape((-1, 1))

    cam_pos = np.concatenate(
        (x_coords, y_coords, np.ones(x_coords.shape) * pos_z), axis=1)
    cam_pos = cam_pos.tolist()

    return cam_pos


def sample_camera_positions_sphere(
        n=1,
        r=1,
        yaw_mean=0.0,
        yaw_std=np.sqrt(np.pi),
        pitch_mean=0.0,
        pitch_std=np.sqrt(np.pi),
        given_yaws=None,
        given_pitches=None,
        flag_rnd=True,
        flag_det_horizontal=True,
        sample_method='uniform',
        n_truncated_stds=2,
        device=torch.device('cpu'),
):
    """Sample points from a sphere's araes defined by ranges of yaw and pitch.

    Note, in this function, we assume the origin locates at the center of sphere and:
    - +X: backward
    - +Y: right
    - +Z: upward

    - yaw is for angle starting from +X on XY-plane with anticlockwise direction
      - positive: right semisphere when facing -X (forward)
      - negative: left semisphere when facing -X (forward)
    - pitch is for angle starting from +X in XZ-plane with anticlockwise direction

    In other words, yaw is for horizontal position while pitch is for vertical poition.

    We define center-facing MPI as yaw=0, pitch=0.
    """

    if given_yaws is None:
        assert given_pitches is None
        if flag_rnd:
            if sample_method == 'uniform':
                yaws = (torch.rand((n, 1), device=device) -
                        0.5) * 2 * n_truncated_stds * yaw_std + yaw_mean
                pitches = (torch.rand((n, 1), device=device) -
                           0.5) * 2 * n_truncated_stds * pitch_std + pitch_mean
            elif sample_method in ['normal', 'gaussian']:
                yaws = torch.randn((n, 1), device=device) * yaw_std + yaw_mean
                pitches = torch.randn(
                    (n, 1), device=device) * pitch_std + pitch_mean
            elif sample_method == 'truncated_gaussian':
                yaws = truncated_normal(
                    torch.zeros((n, 1), device=device),
                    mean=yaw_mean,
                    std=yaw_std,
                    n_truncted_stds=n_truncated_stds,
                )
                pitches = truncated_normal(
                    torch.zeros((n, 1), device=device),
                    mean=pitch_mean,
                    std=pitch_std,
                    n_truncted_stds=n_truncated_stds,
                )
                assert torch.all(
                    yaws >= yaw_mean -
                    n_truncated_stds * yaw_std) and torch.all(
                        yaws <= yaw_mean + n_truncated_stds * yaw_std)
                assert torch.all(
                    pitches >= pitch_mean -
                    n_truncated_stds * pitch_std) and torch.all(
                        pitches <= pitch_mean + n_truncated_stds * pitch_std)
            else:
                raise ValueError
        else:
            if flag_det_horizontal:
                # yaws = torch.linspace(min_yaw, max_yaw, steps=n, device=device).reshape((n, 1))
                yaws = (torch.linspace(
                    -n_truncated_stds,
                    n_truncated_stds,
                    steps=n,
                    device=device).reshape((n, 1))) * yaw_std + yaw_mean
                pitches = torch.ones((n, 1), device=device) * pitch_mean
            else:
                yaws = torch.ones((n, 1), device=device) * yaw_mean
                pitches = (torch.linspace(
                    -n_truncated_stds,
                    n_truncated_stds,
                    steps=n,
                    device=device).reshape((n, 1))) * pitch_std + pitch_mean
    else:
        yaws = given_yaws
        pitches = given_pitches

    # pitches = torch.clamp(pitches, 1e-5, math.pi - 1e-5)

    output_points = torch.zeros((n, 3), device=device)
    output_points[:, 0:1] = r * torch.abs(torch.cos(pitches)) * torch.cos(yaws)
    output_points[:, 1:2] = r * torch.abs(torch.cos(pitches)) * torch.sin(yaws)
    output_points[:, 2:3] = r * torch.sin(pitches)

    # we provide transformation matrix for sphere-based cooridnate system to MPI-based coordinate system.

    return output_points, pitches, yaws


def create_cam2sphere_sys_matrix(forward_vector, origin, device=None):
    """Takes in the direction the camera is pointing and the camera origin and
    returns a cam2sphere_sys matrix.

    NOTE: our camera is defined as: +X: right; +Y: downward; +Z: forward

    modified from https://github.com/marcoamonteiro/pi-GAN/blob/896b8994f415b7efc73f97868a9c99f59ac51adb/generators/volumetric_rendering.py#L176
    For look-at extrinsic matrix, see:
    - https://ksimek.github.io/2012/08/22/extrinsic/
    - https://learnopengl.com/Getting-started/Camera

    Essentially, the look-at matrix directly defines the new basis and serves as a change-of-basis matrix.
    Specifically, assume sphere coordinate system's bases are X = (X1^T, X2^T, X3^T).
    Here X1, X2, X3 are row vectors.
    Now we define new bases Y = (Y1^T, Y2^T, Y3^T) whose origin is the same as X.

    If Y1 = a1_1 X1 + a1_2 X2 + a1_3 X3,
    then Y^T = A X^T.

    Think about a point
    U = u1 X1 + u2 X2 + u3 X3
      = (u1, u2, u3) X^T
      = (u1, u2, u3) A^{-1} Y^T

    So coordinate changes are (u1, u2, u3) A^{-1}.
    Namely, A^{-1} * coord_wrt_X = coord_wrt_Y.
    Look-at camera just defines A, each row of which is a new basis.
    """

    forward_vector = normalize_vecs(forward_vector)

    # +Y: upward
    # NOTE: down_vector is defined in sphere coordinate system.
    # In sphere coordinate system, we have +X: backward, +Y: right, +Z: upward
    down_vector = torch.tensor([0, 0, -1], dtype=torch.float,
                               device=device).expand_as(forward_vector)

    # +X: right
    right_vector = normalize_vecs(
        torch.cross(down_vector, forward_vector, dim=-1))

    down_vector = normalize_vecs(
        torch.cross(forward_vector, right_vector, dim=-1))

    rotation_matrix = torch.eye(
        4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = torch.stack(
        (right_vector, down_vector, forward_vector), axis=-1)

    # tmp = np.array([0, 0, 0, 1]).reshape((-1, 1))
    # print("\nrotation_matrix: ", rotation_matrix, np.matmul(rotation_matrix, tmp))

    translation_matrix = torch.eye(
        4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = origin

    cam2world = translation_matrix @ rotation_matrix

    return cam2world


def create_sphere2world_sys_matrix_for_axis(sphere_center):
    """In this function, we directly compute matrix for coordinate system's
    axis transformation.

    Specifically, we first construct transformation matrix that transforms
    coordinates axises. Then, matrix that transforms the point's coordinates
    are just inverse of the above matrix.
    """

    # NOTE: we do not find the conversion strategy for quaterion to rotation matrix in Scipy.
    # Therefore, we do not know whether it is intrinsic or extrinsic and we cannot guarantee the correctness.
    """
    # scipy's quat is scalar-last format
    # this changes cooridnate system to: +X: right; +Y: forward; +Z: upward
    rot_mat1 = np.eye(4)
    rot_mat1[:3, :3] = Rotation.from_quat(
        [0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)]
    ).as_matrix()
    # this changes cooridnate system to: +X: right; +Y: downward; +Z: forward
    rot_mat2 = np.eye(4)
    rot_mat2[:3, :3] = Rotation.from_quat(
        [np.sin(-1 * np.pi / 4), 0, 0, np.cos(-1 * np.pi / 4)]
    ).as_matrix()
    """
    """
    # scipy's Euler use XYZ for intrinsic and xyz for extrinsic
    # this changes cooridnate system to: +X: right; +Y: forward; +Z: upward
    rot_mat1 = np.eye(4)
    rot_mat1[:3, :3] = Rotation.from_euler(
        "Z", 90, degrees=True
    ).as_matrix()
    # this changes cooridnate system to: +X: right; +Y: downward; +Z: forward
    rot_mat2 = np.eye(4)
    rot_mat2[:3, :3] = Rotation.from_euler(
        "X", -90, degrees=True
    ).as_matrix()

    # NOTE: I have no idea why we must put rot_mat1 first.
    # Otherwise, it will not work.
    rot_mat_for_axis = np.matmul(rot_mat1, rot_mat2)
    """

    rot_mat_for_axis = np.eye(4)
    rot_mat_for_axis[:3, :3] = Rotation.from_euler(
        'ZX', [90, -90], degrees=True).as_matrix()

    translate_mat_for_axis = np.eye(4)
    translate_mat_for_axis[:3, 3] = -1 * sphere_center.reshape(-1)

    transform_mat_for_axis = np.matmul(rot_mat_for_axis,
                                       translate_mat_for_axis)
    transform_mat = np.linalg.inv(transform_mat_for_axis)

    # tmp = np.array([1, 0, 0, 1]).reshape((-1, 1))
    # tmp2 = np.matmul(np.linalg.inv(translate_mat_for_axis), tmp)
    # print("\ntranslate_mat_for_axis: \n", translate_mat_for_axis, "\n", np.linalg.inv(translate_mat_for_axis), "\n", tmp2)
    # print("\nrot_mat_for_axis: ", np.matmul(np.linalg.inv(rot_mat_for_axis), tmp))
    # # print("\nrot1: ", np.matmul(np.linalg.inv(rot_mat1), tmp))  # [1, 0, 0] -> [0, -1, 0]
    # # print("\nrot2: ", np.matmul(np.linalg.inv(rot_mat2), tmp))  # [1, 0, 0] -> [1, 0, 0]
    # print("\ntransform_mat: ", np.matmul(transform_mat, tmp))

    return transform_mat_for_axis


def create_sphere2world_sys_matrix_for_coord(sphere_center):
    """In this function, we directly compute matrix for coordinate
    transformation."""
    """
    # NOTE: this one should also work.
    # However, we do not find the conversion strategy for quaterion to rotation matrix in Scipy.
    # Therefore, we do not know whether it is intrinsic or extrinsic and we cannot guarantee the correctness.

    # scipy's quat is scalar-last format
    # this changes cooridnate system to: +X: right; +Y: forward; +Z: upward
    rot_mat1 = np.eye(4)
    rot_mat1[:3, :3] = Rotation.from_quat(
        [0, 0, np.sin(-np.pi / 4), np.cos(-np.pi / 4)]
    ).as_matrix()
    # this changes cooridnate system to: +X: right; +Y: downward; +Z: forward
    rot_mat2 = np.eye(4)
    rot_mat2[:3, :3] = Rotation.from_quat(
        [np.sin(np.pi / 4), 0, 0, np.cos(np.pi / 4)]
    ).as_matrix()
    """

    # scipy's Euler use XYZ for intrinsic and xyz for extrinsic
    # this changes cooridnate system to: +X: right; +Y: forward; +Z: upward
    rot_mat1 = np.eye(4)
    rot_mat1[:3, :3] = Rotation.from_euler('Z', -90, degrees=True).as_matrix()
    # this changes cooridnate system to: +X: right; +Y: downward; +Z: forward
    rot_mat2 = np.eye(4)
    rot_mat2[:3, :3] = Rotation.from_euler('X', 90, degrees=True).as_matrix()

    rot_mat = np.matmul(rot_mat2, rot_mat1)

    translate_mat = np.eye(4)
    translate_mat[:3, 3] = sphere_center.reshape(-1)

    transform_mat_for_coord = np.matmul(translate_mat, rot_mat)

    transform_mat = transform_mat_for_coord

    # tmp = np.array([1, 0, 0, 1]).reshape((-1, 1))
    # tmp2 = np.matmul(translate_mat, tmp)
    # print("\ntranslate_mat: \n", translate_mat, tmp2)
    # print("\nrot_mat: ", np.matmul(rot_mat, tmp))
    # print("\ntransform_mat: ", np.matmul(transform_mat, tmp))

    return transform_mat, rot_mat


def gen_sphere_path(
        n_cams: int,
        sphere_center: np.ndarray,
        sphere_r: Union[float, None],
        # yaw_range=[-np.pi, np.pi],
        # pitch_range=[0.0, np.pi],
        yaw_mean=0.0,
        yaw_std=np.sqrt(np.pi),
        pitch_mean=0.0,
        pitch_std=np.sqrt(np.pi),
        given_yaws=None,
        given_pitches=None,
        flag_rnd=True,
        flag_det_horizontal=True,
        sample_method='uniform',
        n_truncated_stds=2,
        device=torch.device('cpu'),
):
    """Generate random cameras on the sphere.

    Args:
        n_cams (int): #cameras
        sphere_center (np.ndarray): coordinates of the center of sphere.
            The coordinates are defined in MPI-based coordinate system.
        yaw_range (list, optional): Defaults to [-np.pi, np.pi].
        pitch_range (list, optional): Defaults to [0.0, np.pi].
        device ([type], optional): Defaults to torch.device("cpu").
    """
    if sphere_r is None:
        sphere_r = np.linalg.norm(sphere_center, ord=2)

    # NOTE: here camera positions are parameterized in sphere-based coordinate system
    # [#cam, 3]
    cam_pos_in_sphere_sys, pitches, yaws = sample_camera_positions_sphere(
        n=n_cams,
        r=sphere_r,
        # yaw_range=yaw_range,
        # pitch_range=pitch_range,
        yaw_mean=yaw_mean,
        yaw_std=yaw_std,
        pitch_mean=pitch_mean,
        pitch_std=pitch_std,
        given_yaws=given_yaws,
        given_pitches=given_pitches,
        flag_rnd=flag_rnd,
        flag_det_horizontal=flag_det_horizontal,
        sample_method=sample_method,
        n_truncated_stds=n_truncated_stds,
        device=device,
    )

    # [#cam, 3]
    forward_vec_sphere_sys = normalize_vecs(-cam_pos_in_sphere_sys)

    # [#cam, 4, 4]
    cam2sphere_sys_mat = (
        create_cam2sphere_sys_matrix(
            forward_vec_sphere_sys, cam_pos_in_sphere_sys,
            device=device).cpu().numpy())

    # NOTE: we need to transform all camera positions back to MPI's coordinate system that:
    # - for sphere-based coordinate system: - +X: backward, +Y: right, +Z: upward, origin is at sphere's center
    # - for MPI-based coordinate system: +X: right, +Y: downward, +Z: forward, origin is not at sphere's center
    """
    # NOTE: this one also works
    # transform axis first, then coordinates
    transform_mat_for_axis = create_sphere2world_sys_matrix_for_axis(sphere_center)
    transform_mat = np.linalg.inv(transform_mat_for_axis)
    """

    # directly transform coordinates
    transform_mat, pure_rot_mat = create_sphere2world_sys_matrix_for_coord(
        sphere_center)

    # [B, 4, 4]
    cam2mpi_sys_mat = np.matmul(transform_mat, cam2sphere_sys_mat)

    # # NOTE: this sphere_sys has same coord axis orientations as mpi, +X: right, +Y: downward, +Z: forward
    # cam2sphere_sys_mat2 = np.matmul(pure_rot_mat, cam2sphere_sys_mat)

    # print("\ncam2sphere_sys_mat: ", cam2sphere_sys_mat, "\n")
    # print("\ntransform_mat: ", transform_mat, "\n")
    # print("\ncam2mpi_sys_mat ", cam2mpi_sys_mat)

    # pred_yaws, pred_pitches = compute_pitch_yaw_from_w2c_mat(torch.inverse(torch.FloatTensor(cam2mpi_sys_mat)), torch.FloatTensor(sphere_center))
    # print("\nyaw: ", torch.cat((torch.FloatTensor(yaws), pred_yaws), dim=1), "\n")
    # print("\npitch: ", torch.cat((torch.FloatTensor(pitches), pred_pitches), dim=1), "\n")

    return cam2mpi_sys_mat, yaws, pitches


def compute_w2c_mat_from_estimated_pose_ffhq(angles,
                                             trans,
                                             sphere_center,
                                             sphere_r=1.0,
                                             normalize_trans=False,
                                             device=torch.device('cpu')):
    """
    Return:
        rot     -- torch.tensor, size (B, 3, 3) pts @ trans_mat

    Parameters:
        angles  -- torch.tensor, size (B, 3), radian
        trans   -- torch.tensor, size (B, 3)

    The goal of this function is to produce a world2cam matrix
    that can transform MPI's volume into Deep3DFace's camera coordinate system.
    Procedures are following:
    1) Remember for MPI's world coord system, MPI's center is at (0, 0, sphere_center).
       We move MPI's center to the origin (0, 0, 0);
    2) MPI's world coord system uses +X right, +Y downward, +Z forward while Deep3DFace uses +X right, +Y up, +Z backward.
       We need to rorate MPI to align with Deep3DFace.
    3) Use Deep3DFace's estimated pose to transform MPI into camera's coord system.
       Note, currently, camera coord sys still uses +X right, +Y up, +Z backward.
    4) We need to rorate the MPI again to align with our definition +X right, +Y downward, +Z forward.

    Ref:
    - https://github.com/sicxu/Deep3DFaceRecon_pytorch/blob/091e84551a6f78cc5046320b5229fde2e5e902c1/models/bfm.py#L174
    - https://github.com/sicxu/Deep3DFaceRecon_pytorch/blob/091e84551a6f78cc5046320b5229fde2e5e902c1/models/bfm.py#L210

    NOTE, the world coordinate system is defined as +X right, +Y up, +Z backward
    - https://github.com/microsoft/Deep3DFaceReconstruction/blob/1092522/readme.md#note
    - https://nvlabs.github.io/nvdiffrast/#coordinate-systems
    """

    # Question: camera distance 1.0 or 10.0?
    # NOTE: It seems like it must be 10.0.
    camera_distance = 10.0

    # angles = torch.zeros(angles.shape, device=angles.device)

    batch_size = angles.shape[0]

    # procedure 1
    # make MPI's volume's center to the origin
    t_mat1 = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    t_mat1[:, 2, 3] = -1 * sphere_center

    transform_mat = t_mat1

    # procedure 2
    # transform from MPI (+X right, +Y downward, +Z forward) to Deep3DFace (+X right, +Y up, +Z backward)
    rot_mat1 = torch.eye(
        4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    rot_mat1[:, 1, 1] = -1
    rot_mat1[:, 2, 2] = -1

    transform_mat = torch.matmul(rot_mat1, transform_mat)

    # procedure 3
    ones = torch.ones([batch_size, 1]).to(device)
    zeros = torch.zeros([batch_size, 1]).to(device)
    x, y, z = (
        angles[:, :1],
        angles[:, 1:2],
        angles[:, 2:],
    )

    rot_x = torch.cat([
        ones, zeros, zeros, zeros,
        torch.cos(x), -torch.sin(x), zeros,
        torch.sin(x),
        torch.cos(x)
    ],
                      dim=1).reshape([batch_size, 3, 3])

    rot_y = torch.cat([
        torch.cos(y), zeros,
        torch.sin(y), zeros, ones, zeros, -torch.sin(y), zeros,
        torch.cos(y)
    ],
                      dim=1).reshape([batch_size, 3, 3])

    rot_z = torch.cat([
        torch.cos(z), -torch.sin(z), zeros,
        torch.sin(z),
        torch.cos(z), zeros, zeros, zeros, ones
    ],
                      dim=1).reshape([batch_size, 3, 3])

    rot = rot_z @ rot_y @ rot_x
    # NOTE: in original code, they use x @ R. However, we use R @ x
    # rot = rot.permute(0, 2, 1)

    # NOTE: this matrix transform face's vertices wrt world coordinate system
    tmp_transform_mat = torch.eye(
        4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    tmp_transform_mat[:, :3, :3] = rot
    tmp_transform_mat[:, :3, 3] = trans

    transform_mat = torch.matmul(tmp_transform_mat, transform_mat)

    # we need to translate face's vertices to camera coordinate system, still +X right, +Y up, +Z backward
    t_mat2 = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    t_mat2[:, 2, 3] = -1 * camera_distance

    transform_mat = torch.matmul(t_mat2, transform_mat)

    # procedure 4
    # transform from Deep3DFace (+X right, +Y up, +Z backward) to MPI (+X right, +Y downward, +Z forward)
    transform_mat = torch.matmul(rot_mat1, transform_mat)

    if normalize_trans:
        # [B, 3, 1]
        tmp_rot = transform_mat[:, :3, :3]
        tmp_trans = transform_mat[:, :3, 3:]
        # R @ x + t = 0 -> x = -R^{-1} @ t, [B, 3, 1]
        tmp_cam_center_in_world = -1 * torch.matmul(
            torch.inverse(tmp_rot), tmp_trans)
        # [1, 3, 1]
        sphere_center_coord = torch.FloatTensor([0, 0, sphere_center],
                                                device=transform_mat.device)
        # [B, 3, 1]
        sphere_center_coord = sphere_center_coord.reshape(
            (1, 3, 1)).expand(tmp_trans.shape[0], -1, -1)
        dist_vec = tmp_cam_center_in_world - sphere_center_coord
        # [B, 1, 1]
        tmp_trans_norm = torch.norm(dist_vec, p=2, dim=1, keepdim=True)
        # [B, 3]
        tmp_new_cam_center_in_world = dist_vec / tmp_trans_norm * sphere_r + sphere_center_coord
        tmp_new_trans = -1 * torch.matmul(tmp_rot, tmp_new_cam_center_in_world)
        transform_mat[:, :3, 3:] = tmp_new_trans

    return transform_mat


def compute_w2c_mat_from_estimated_pose_afhq(c2w_mats,
                                             sphere_center,
                                             sphere_r=1.0,
                                             normalize_trans=False,
                                             device=torch.device('cpu')):
    """
    Return:
        rot     -- torch.tensor, size (B, 3, 3) pts @ trans_mat

    Parameters:
        angles  -- torch.tensor, size (B, 3), radian
        trans   -- torch.tensor, size (B, 3)

    The goal of this function is to produce a world2cam matrix
    that can transform MPI's volume into camera coordinate system.
    Procedures are following:
    1) Remember for MPI's world coord system, MPI's center is at (0, 0, sphere_center).
       We move MPI's center to the origin (0, 0, 0);
    2) MPI's world coord system uses +X right, +Y downward, +Z forward while OpenCV uses +X right, +Y up, +Z backward for world.
       We need to rorate MPI to align with OpenCV.
    3) Use PnP's estimated pose to transform MPI into camera's coord system.
       Note, currently, camera coord sys uses +X right, +Y down, +Z forward
       (see https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html).
    """

    batch_size = c2w_mats.shape[0]

    # procedure 1
    # make MPI's volume's center to the origin
    t_mat1 = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    t_mat1[:, 2, 3] = -1 * sphere_center

    transform_mat = t_mat1

    # procedure 2
    # transform from MPI (+X right, +Y downward, +Z forward) to OpenCV's world (+X right, +Y up, +Z backward)
    rot_mat1 = torch.eye(
        4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    rot_mat1[:, 1, 1] = -1
    rot_mat1[:, 2, 2] = -1

    transform_mat = torch.matmul(rot_mat1, transform_mat)

    # procedure 3
    transform_mat = torch.matmul(torch.inverse(c2w_mats), transform_mat)

    if normalize_trans:
        # [B, 3, 1]
        tmp_rot = transform_mat[:, :3, :3]
        tmp_trans = transform_mat[:, :3, 3:]
        # R @ x + t = 0 -> x = -R^{-1} @ t, [B, 3, 1]
        tmp_cam_center_in_world = -1 * torch.matmul(
            torch.inverse(tmp_rot), tmp_trans)
        # [1, 3, 1]
        sphere_center_coord = torch.FloatTensor([0, 0, sphere_center],
                                                device=transform_mat.device)
        # [B, 3, 1]
        sphere_center_coord = sphere_center_coord.reshape(
            (1, 3, 1)).expand(tmp_trans.shape[0], -1, -1)
        dist_vec = tmp_cam_center_in_world - sphere_center_coord
        # [B, 1, 1]
        tmp_trans_norm = torch.norm(dist_vec, p=2, dim=1, keepdim=True)
        # [B, 3]
        tmp_new_cam_center_in_world = dist_vec / tmp_trans_norm * sphere_r + sphere_center_coord
        tmp_new_trans = -1 * torch.matmul(tmp_rot, tmp_new_cam_center_in_world)
        transform_mat[:, :3, 3:] = tmp_new_trans

    return transform_mat


def compute_pitch_yaw_from_w2c_mat(w2c_mat, sphere_c):
    """The goal of this function is to return camera's pitch and yaw defined in
    func `sample_camera_positions_sphere` from world2camera transformation
    matrix.

    NOTE, `sample_camera_positions_sphere` defines +X backward, +Y right, +Z upward,
    while world coordinate system is +X right, +Y downward, +Z forward.

    Copy from `sample_camera_positions_sphere`:
    - yaw is for angle starting from +X on XY-plane with anticlockwise direction
      - positive: right semisphere when facing -X (forward)
      - negative: left semisphere when facing -X (forward)
    - pitch is for angle starting from +X in XZ-plane with anticlockwise direction
    """
    assert sphere_c.ndim <= 2, f'{sphere_c.shape}'
    assert w2c_mat.ndim == 3, f'{w2c_mat.shape}'

    bs = w2c_mat.shape[0]

    # 1) get world2sphere transformation matrix
    sphere2world_mat, pure_rot_mat = create_sphere2world_sys_matrix_for_coord(
        sphere_c.numpy())
    # [4, 4]
    world2sphere_mat = torch.inverse(torch.FloatTensor(sphere2world_mat))
    # [B, 4, 4]
    world2sphere_mat = world2sphere_mat.unsqueeze(0).expand(bs, -1, -1)

    # 2) get camera backward direction in sphere coorindate system
    # [B, 4, 1]
    cam_origin = torch.FloatTensor([0, 0, 0, 1]).reshape(
        (1, 4, 1)).expand(bs, -1, -1)
    cam_pos_in_world = torch.matmul(torch.inverse(w2c_mat), cam_origin)
    # [B, 3, 1]
    cam_pos_in_sphere = torch.matmul(world2sphere_mat, cam_pos_in_world)[:, :3]
    # [B, 3, 1]
    cam_pos_in_sphere = cam_pos_in_sphere / torch.norm(
        cam_pos_in_sphere, p=2, dim=1, keepdim=True)

    xs_in_sphere = cam_pos_in_sphere[:, 0]
    ys_in_sphere = cam_pos_in_sphere[:, 1]
    zs_in_sphere = cam_pos_in_sphere[:, 2]

    # 3) compute yaws
    yaws = torch.atan2(ys_in_sphere, xs_in_sphere)

    # 4) compute pitch
    # NOTE: torch.acos(zs_in_sphere) returns angle starting from +Z in XZ-plane with clockwise direction
    pitches = np.pi / 2 - torch.acos(zs_in_sphere)

    return yaws, pitches
