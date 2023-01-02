#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import copy
import glob
import os
from typing import Dict, List, Union

import numpy as np
import torch
import tqdm
from PIL import Image

from .cam_utils import gen_sphere_path
from .camera import Camera
from .color_grad import RGB_to_hex, linear_gradient, rgb_from_color_dict


def sample_distance(dmin: float, dmax: float, num_samples: int, method: str,
                    **kwargs) -> List:
    """Sampling of distances in the range [dmin, dmax]

    Args:
        dmin (float): minimum radius
        dmax (float): maximum radius
        num_samples (int): number of samples
        sampling (str): sampling method ("uniform", "sqrt", "squared", "inverse", "deepview")

    Returns:
        List: list of radii of the spheres
    """
    # This leads to large increments in close range and small increments in far range, which is undesirable.
    assert 0 < dmin <= dmax
    assert 1 <= num_samples < 9999

    if method == 'uniform':
        radii = [i for i in np.linspace(dmin, dmax, num=num_samples)]
    elif method == 'log-uniform':
        radii = np.exp(
            np.linspace(np.log(dmin), np.log(dmax), num=num_samples)).tolist()
    elif method == 'sqrt':
        radii = [
            i**2 for i in np.linspace(dmin**0.5, dmax**0.5, num=num_samples)
        ]
    elif method == 'squared':
        # This leads to large increments in close range and small increments in far range, which is undesirable.
        radii = [
            np.sqrt(i) for i in np.linspace(dmin**2, dmax**2, num=num_samples)
        ]
    elif method == 'inverse':
        # This leads to highly sparse sampling at far distances
        radii = [
            1 / i for i in np.linspace(1 / dmax, 1 / dmin, num=num_samples)
        ]
        radii.reverse()
    else:
        raise ValueError

    return np.array(radii, dtype=np.float32)


def gen_tex_3d_coord_grid(
        res_spatial: List[float],
        res_tex: List[int],
        distance: float,
        device: torch.device = torch.device('cpu'),
):
    """[summary]

    Args:
        res_spatial (List[float]): spatial resolution in 3D, (h, w)
        res_tex (List[int]): texture plane resolution, (h, w)
        distance (float): [description]
        device (torch.device, optional): [description]. Defaults to torch.device("cpu").

    Returns:
        [type]: [description]
    """
    h_spatial, w_spatial = res_spatial
    h_tex, w_tex = res_tex

    yy = torch.arange(start=-1, end=1, step=2.0 / h_tex, device=device)
    xx = torch.arange(start=-1, end=1, step=2.0 / w_tex, device=device)
    yy, xx = torch.meshgrid(yy, xx)

    # un-normalize from [-1, 1] to x and y
    x = xx * w_spatial / 2
    y = yy * h_spatial / 2

    # convert to xyz
    z = torch.ones(x.shape, device=device) * distance

    # return pointcloud
    xyz = torch.stack((x, y, z), dim=0)

    return xyz.view([1, 3, h_tex, w_tex])


def mpi_from_content_imgs(
    *,
    tex_h: int,
    tex_w: int,
    content_rgbas: List[Union[None, np.ndarray]],
    # fill_colors: List[Union[None, np.ndarray]],
    # content_scales: List[float],
    content_hws: List[Union[None, List[int]]],
    spatial_hws: List[Union[None, List[float]]] = [None],
    spatial_fovs: List[Union[None, float]] = [None],
    spatial_h_scales: List[Union[float, None]] = [None],
    spatial_aspect_ratios: List[Union[float, None]] = [None],
    plane_fixed_pos: List[Union[None, Dict[str, int]]] = [None],
    plane_rnd_pos: List[bool] = [False],
    dmin: float = 1.0,
    dmax: float = 10.0,
    fixed_distances: Union[None, List[float]] = None,
    content_resize_already: bool = False,
    return_fg_range: bool = True,
    device: torch.device = torch.device('cpu'),
) -> Union[torch.tensor, List[int], List[List[int]], List[List[int]]]:
    """Generate MPI from a pair of background and foreground images.

    Args:
        spatial_h_scale (float): relative scale wrt plance's depth
        spatial_aspect_ratio (float): plane's spatial aspect ratio
        tex_h (int), tex_w (int): resolution of plane
        fg_rgba (np.ndarray): foreground RGB-alpha, range [0, 255]
        bg_rgb (np.ndarray): backgroun RGB, range [0, 255]
        bg_alpha (np.ndarray): background alpha, optional
        fg_scale (float): foreground's scale wrt texture resolution
        fg_pos (Dict[str, int]): specify foreground's position
        fg_fill_color (np.ndarray): color for foreground's shape
    """

    for elem in content_rgbas:
        assert elem.shape[
            2] == 4, f'Images for plane must be RGB-alpha format. However, we receive {elem.shape}.'

    assert len(content_rgbas) == len(
        content_hws), f'{len(content_rgbas)}, {len(content_hws)}'
    assert len(content_rgbas) == len(
        spatial_hws), f'{len(content_rgbas)}, {len(spatial_hws)}'
    if fixed_distances is not None:
        assert len(content_rgbas) == len(
            fixed_distances), f'{len(content_rgbas)}, {len(fixed_distances)}'

    n_planes = len(content_rgbas)

    all_plane_rgbas = []

    for i in range(n_planes):

        # we assume the RGB-alphas are stored in the order of front-to-back

        if content_resize_already:
            cur_rgba = content_rgbas[i]
        else:
            cur_rgba = copy.deepcopy(content_rgbas[i])
        # cur_row, cur_col, _ = np.where(fg_rgba[..., 3:] > 0)
        # fg_rgba[tmp_row, tmp_col, :3] = fg_fill_color.reshape((1, 1, 3))

        if isinstance(cur_rgba, torch.Tensor):
            assert content_resize_already

        # content_h = int(tex_h * content_scales[i])
        # content_w = int(tex_w * content_scales[i])
        content_h, content_w = content_hws[i]
        if i == 0:
            fg_tex_h = content_h
            fg_tex_w = content_w

        if plane_fixed_pos[i] is not None:
            start_row = plane_fixed_pos[i]['center_row'] - content_h // 2
            start_col = plane_fixed_pos[i]['center_col'] - content_w // 2
            assert (start_row >= 0) and (start_col >=
                                         0), f'{start_row}, {start_col}'
        else:
            if plane_rnd_pos[i]:
                start_row = np.random.randint(0, tex_h - content_h)
                start_col = np.random.randint(0, tex_w - content_w)
            else:
                start_row = (tex_h - content_h) // 2
                start_col = (tex_w - content_w) // 2
        end_row = start_row + content_h
        end_col = start_col + content_w

        if isinstance(cur_rgba, torch.Tensor):
            plane_rgba = torch.zeros((tex_h, tex_w, 4),
                                     dtype=torch.uint8,
                                     device=cur_rgba.device)
        else:
            plane_rgba = np.zeros((tex_h, tex_w, 4), dtype=np.uint8)

        if content_resize_already:
            plane_rgba[start_row:end_row, start_col:end_col, :] = cur_rgba
        else:
            plane_rgba[start_row:end_row, start_col:end_col, :] = np.array(
                Image.fromarray(cur_rgba).resize((content_w, content_h),
                                                 resample=Image.LANCZOS))

        if isinstance(plane_rgba, torch.Tensor):
            all_plane_rgbas.append(plane_rgba.float() / 255.0)
        else:
            all_plane_rgbas.append(plane_rgba.astype(np.float32) / 255.0)

    if fixed_distances is None:
        distances = sample_distance(
            dmin=dmin, dmax=dmax, num_samples=n_planes, method='inverse')
    else:
        distances = fixed_distances
    distances = np.sort(distances)

    # Create a list of planes
    planes = []
    dhws = []

    for i, d in enumerate(distances):
        # Set the spatial dimension of a plane in the world space
        # NOTE: these numbers are arbitrary

        # Create a plane texture at distance d along the z axis
        if spatial_hws[i] is None:
            spatial_h, spatial_w = (
                spatial_h_scales[i] * d,
                spatial_h_scales[i] * spatial_aspect_ratios[i] * d,
            )
        else:
            spatial_h, spatial_w = spatial_hws[i]

        _plane = all_plane_rgbas[i]
        if not isinstance(_plane, torch.Tensor):
            _plane = torch.FloatTensor(all_plane_rgbas[i])
        # [h, w, 4] -> [1, 4, h, w]
        _plane = _plane.permute(2, 0, 1).unsqueeze(0).to(device)
        # _plane = all_plane_rgbas[i][np.newaxis, ...]

        dhws.append((d, spatial_h, spatial_w))
        planes.append(_plane)

    # [#planes, 4, H, W]
    planes = torch.cat(planes, dim=0)
    # planes = torch.from_numpy(np.concatenate(planes, axis=0)).permute(0, 3, 1, 2).to(device)
    # [#planes, 3]
    dhws = torch.tensor(dhws)

    if return_fg_range:
        # we find foreground's non-zeros alpha positions
        tmp_row, tmp_col, _ = np.where(all_plane_rgbas[0][..., 3:] > 0)
        fg_range = {
            'fg_tex_h': fg_tex_h,
            'fg_tex_w': fg_tex_w,
            'min_row': int(np.min(tmp_row)),
            'max_row': int(np.max(tmp_row)),
            'min_col': int(np.min(tmp_col)),
            'max_col': int(np.max(tmp_col)),
        }
    else:
        fg_range = None

    return planes, dhws, fg_range


def get_vis_bound_in_pix_coords_from_xyz(
    spatial_h: float,
    spatial_w: float,
    vis_bound_xyz: Dict[str, float],
    tex_h: int,
    tex_w: int,
    content_h: int,
    content_w: int,
):
    """This function computes boundary for visible area in pixel coordinates
    (row/col) from boundary in 3D xyz coordinates.

    Args:
        spatial_h (float): height for plane centered at (x = 0, y = 0)
        spatial_w (float): width for plane centered at (x = 0, y = 0)
        vis_bound_xyz (Dict[str, float]): boundary for visible area in xyz coordinates
        tex_h (int): plane's texture height
        tex_w (int): plane's texture weight
        content_h (int): plane's content height, content is the image to be stored in that plane.
            Height of content is differenet from height of the texture itself.
            Actuall, content_h <= tex_h.
        content_w (int): plane's content weight
    """
    # +Y points downward, top-most position has smallest value
    range_h = [-1 * spatial_h / 2, spatial_h / 2]
    cell_h = spatial_h / tex_h
    # +X points right, left-most position has smallest value
    range_w = [-1 * spatial_w / 2, spatial_w / 2]
    cell_w = spatial_w / tex_w

    # NOTE: pay attention to the floor/ceil here. We want to have a compact area.
    min_row = max(0,
                  int(np.ceil((vis_bound_xyz['min_y'] - range_h[0]) / cell_h)))
    max_row = min(
        tex_h - 1,
        int(np.floor((vis_bound_xyz['max_y'] - range_h[0]) / cell_h)))
    min_col = max(0,
                  int(np.ceil((vis_bound_xyz['min_x'] - range_w[0]) / cell_w)))
    max_col = min(
        tex_w - 1,
        int(np.floor((vis_bound_xyz['max_x'] - range_w[0]) / cell_w)))

    # we store the range for center pixel of content
    vis_bound_pix = {
        'min_row': min(tex_h - 1, int(min_row + content_h // 2)),
        'max_row': max(0, int(max_row - content_h // 2)),
        'min_col': min(tex_w - 1, int(min_col + content_w // 2)),
        'max_col': max(0, int(max_col - content_w // 2)),
    }

    assert (vis_bound_pix['min_row'] < vis_bound_pix['max_row']
            ), f"{vis_bound_pix['min_row']}, {vis_bound_pix['max_row']}"
    assert (vis_bound_pix['min_col'] < vis_bound_pix['max_col']
            ), f"{vis_bound_pix['min_col']}, {vis_bound_pix['max_col']}"

    return vis_bound_pix


def mpi_from_plane_imgs(dmin: float = 1.0,
                        dmax: float = 10.0,
                        plane_rgbas: List[np.ndarray] = []):
    """Generate MPI from a list of RGB-alpha.

    Args:
        dmin (int), dmax (int): depth boundaries. When sampling with 'inverse', these two are essentially disparities.
        plane_rgbas (List[np.ndarray]): list of plane images (RGBA, np.uint8), stored from back to front.
            Namely, 1st elem is the furthest plane.

    Returns:
        mpi (MPI)
    """

    for tmp in plane_rgbas:
        assert tmp.dtype == np.uint8

    n_planes = len(plane_rgbas)

    distances = sample_distance(
        dmin=dmin, dmax=dmax, num_samples=n_planes, method='inverse')

    # make planes stored as from front to back
    plane_rgbas.reverse()

    # Create a list of planes
    planes = []
    dhws = []

    for i, d in enumerate(np.sort(distances)):

        # Set the spatial dimension of a plane in the world space
        # these numbers are arbitrary
        spatial_h, spatial_w = d, 1.3 * d
        dhws.append((d, spatial_h, spatial_w))

        # uint8 -> float32
        rgb_init = plane_rgbas[i][..., :3].astype(np.float32) / 255.0
        alpha_init = plane_rgbas[i][..., 3:].astype(np.float32) / 255.0

        # [h, w, 3] -> [1, 3, h, w]
        rgb_init = torch.FloatTensor(rgb_init).permute(2, 0, 1).unsqueeze(0)
        alpha_init = torch.FloatTensor(alpha_init).permute(2, 0,
                                                           1).unsqueeze(0)

        rgba = torch.cat((rgb_init, alpha_init), dim=1)
        planes.append(rgba)

    planes = torch.cat(planes, dim=0)
    dhws = torch.tensor(dhws)

    tmp_row, tmp_col, _ = np.where(plane_rgbas[0][..., 3:] > 0)
    fg_range = {
        'min_row': int(np.min(tmp_row)),
        'max_row': int(np.max(tmp_row)),
        'min_col': int(np.min(tmp_col)),
        'max_col': int(np.max(tmp_col)),
    }

    return planes, dhws, fg_range


def add_color_to_obj(
    obj_img: np.ndarray,
    base_rgb: np.ndarray,
    grad_color_ratio_between_end_points: float,
    n_grad_color: int,
    light_color_on_top: bool,
):
    """This function add gradient ramp color to objects.

    Args:
        obj_img (np.ndarray): RGBA image
        base_rgb (np.ndarray): when conduct gradually-changed color, base_rgb is the end point for light color
        grad_color_ratio_between_end_points (float): grad_color_ratio_between_end_points * base_rgb is the end point for dark color
        n_grad_color (int): #color points for gradual change
        light_color_on_top (bool): whether to put light color on top
    """
    assert (obj_img.ndim == 3) and (obj_img.shape[-1] == 4)

    end_rgb = (base_rgb * grad_color_ratio_between_end_points).astype(np.uint8)

    start_hex = RGB_to_hex(base_rgb)
    end_hex = RGB_to_hex(end_rgb)

    linear_color_dict = linear_gradient(start_hex, end_hex, n=n_grad_color)
    # [3, N]
    # NOTE: base_rgb is lighter than end_rgb
    linear_colors = rgb_from_color_dict(linear_color_dict)
    if not light_color_on_top:
        linear_colors = linear_colors[:, ::-1]

    obj_alpha = obj_img[..., 3]
    non_zero_rows, non_zero_cols = np.where(obj_alpha > 0)
    min_row = np.min(non_zero_rows)
    max_row = np.max(non_zero_rows)

    row_range = max_row - min_row + 1
    if row_range > n_grad_color:
        n_repeat = int(np.ceil(row_range / n_grad_color))
        linear_colors = np.repeat(linear_colors, n_repeat, axis=1)

    for i in range(row_range):
        tmp_row = min_row + i
        tmp_cols = np.where(obj_alpha[tmp_row, :] > 0)[0]
        # [#nonzero, 3]
        obj_img[tmp_row, tmp_cols, :3] = linear_colors[:, i].reshape((1, 3))

    return obj_img


def compute_rect_overlap_area(a: List[Union[int, float]],
                              b: List[Union[int, float]]):
    """This function computes the overlap area for two rectangles a and b.

    Args:
        a, b: two rectangels, [min_row, max_row, min_col, max_col, area]
            min_row/min_col are included while max_row/max_col are excluded.
    """
    min_row_a, max_row_a, min_col_a, max_col_a, _ = a
    min_row_b, max_row_b, min_col_b, max_col_b, _ = b

    drow = min(max_row_a, max_row_b) - max(min_row_a, min_row_b)
    dcol = min(max_col_a, max_col_b) - max(min_col_a, min_col_b)

    if (drow >= 0) and (dcol >= 0):
        return drow * dcol
    else:
        return 0.0


def sample_rect_pos(
    img_h: int,
    img_w: int,
    cur_h: int,
    cur_w: int,
    prev_rects: List[Union[int, float]],
    obj_max_overlap_thresh: float = 0.5,
):
    """Sample rectangle's positions such that the position's overlap with
    existed rectangles is lower than some threshold.

    Args:
        img_h, img_w: the canvas's size
        cur_h, cur_w: the size of the rectangle to be placed
        prev_rects: list of existed rectangle's information.
            Each element in the list has format [min_row, max_row, min_col, max_col, area]
        obj_max_overlap_thresh: the maximum allowed threshold for overlap ratio.
    """
    max_valid_row = img_h - cur_h + 1
    max_valid_col = img_w - cur_w + 1

    stop_flag = False

    area = cur_h * cur_w

    while not stop_flag:
        # continus until find a position does not exceed the overlap threshold
        cur_min_row = np.random.randint(0, max_valid_row)
        cur_min_col = np.random.randint(0, max_valid_col)
        cur_max_row = cur_min_row + cur_h
        cur_max_col = cur_min_col + cur_w
        cur_rect = [cur_min_row, cur_max_row, cur_min_col, cur_max_col, area]

        if len(prev_rects) > 0:
            for i, tmp_prev_rect in enumerate(prev_rects):
                overlap_area = compute_rect_overlap_area(
                    cur_rect, tmp_prev_rect)
                if (overlap_area > obj_max_overlap_thresh * area) or (
                        overlap_area > tmp_prev_rect[-1] * area):
                    break

            if i == len(prev_rects) - 1:
                stop_flag = True
        else:
            stop_flag = True

    return cur_rect


def gen_plane_imgs_from_objs(
    *,
    tex_h: int,
    tex_w: int,
    obj_img_dir: str,
    n_objs: int,
    n_planes: int,
    obj_max_scale_wrt_canvas: float = 0.8,
    obj_max_overlap_thresh: float = 0.5,
    grad_color_ratio_between_end_points: float = 0.5,
    n_grad_color: int = 100,
    light_color_on_top: bool = True,
):
    """Generate planes images from a set of objects.

    Args:
        tex_h, tex_w (int): canvas's resolution
        obj_img_dir (str): directory for finding objects's images
        n_objs (int): the number of objects to be put on canvas
        n_planes (int): the number of planes to be constructed
        obj_max_scale_wrt_canvas (float): if rescale the object's image,
            this controls the maximum resolution of a single object.
        obj_max_overlap_thresh (flat): maximum overlap ratio between two objects.
            Currently, the algorithm is purely heuristics.
            TODO: change to bin-packing if needed.
        grad_color_ratio_between_end_points (float): for gradient ramp,
            this controls the color change ratio between two end point's colors.
        n_grad_color (int): the number of gradually-changed colors for gradient ramp
        light_color_on_top (bool): whether to put the light end of gradient ramp on object's top.
    """

    all_fs = list(glob.glob(os.path.join(obj_img_dir, '*.png')))

    sampled_idxs = np.random.choice(
        len(all_fs), size=n_objs, replace=True).tolist()
    sampled_fs = [all_fs[_] for _ in sampled_idxs]

    all_objs = [np.array(Image.open(_)) for _ in sampled_fs]

    # put object onto the image plane
    obj_rects = []

    # this cooresponds to span all objects in a square without overlap
    min_scale = 1.0 / np.sqrt(n_objs)
    max_scale = obj_max_scale_wrt_canvas

    all_resized_objs = []

    for obj in all_objs:

        tmp_scale = np.random.uniform(min_scale, max_scale)

        tmp_h, tmp_w, _ = obj.shape

        # these are maximum valid resolution
        max_target_h = int(tex_h * tmp_scale)
        max_target_w = int(tex_w * tmp_scale)

        # we fix the object image's aspect ratio
        target_h = max_target_h
        target_w = int(tmp_w / tmp_h * target_h)
        if target_w > max_target_w:
            target_w = max_target_w
            target_h = int(tmp_h / tmp_w * target_w)

        resized_obj = np.array(
            Image.fromarray(obj).resize((target_w, target_h),
                                        resample=Image.LANCZOS))
        all_resized_objs.append(resized_obj)

        tmp_rect = sample_rect_pos(
            tex_h,
            tex_w,
            target_h,
            target_w,
            obj_rects,
            obj_max_overlap_thresh=obj_max_overlap_thresh,
        )
        obj_rects.append(tmp_rect)

    # assign objects to different planes
    # NOTE: directly random assignment may result in no objects appearing on some specific planes.
    # Therefore, we manually assign them.
    obj_plane_idxs = np.zeros(n_objs)
    n_obj_per_plane = (n_objs - 1) // (n_planes - 1)
    for i in range(1, n_planes):
        start_obj_idx = 1 + (i - 1) * n_obj_per_plane
        end_obj_idx = min(start_obj_idx + n_obj_per_plane, n_objs)
        obj_plane_idxs[start_obj_idx:end_obj_idx] = i

    if end_obj_idx < n_objs:
        tmp_n = n_objs - end_obj_idx
        obj_plane_idxs[end_obj_idx:] = np.random.choice(
            np.arange(1, n_planes, tmp_n))

    # only first object goes to foreground
    assert np.sum(obj_plane_idxs == 0) == 1

    # sample a base color based on plane index.
    # the closer the plane is to the camera, the more closer the base color is to black
    MAX_START_RGB_VAL = 200

    for i in range(n_objs):
        tmp_obj = all_resized_objs[i]
        tmp_plane_idx = obj_plane_idxs[i]

        tmp_start_rgb_val = 128 + int(tmp_plane_idx *
                                      (MAX_START_RGB_VAL - 128) / n_planes)
        tmp_base_rgb = np.array((
            np.random.randint(tmp_start_rgb_val, 256),
            np.random.randint(tmp_start_rgb_val, 256),
            np.random.randint(tmp_start_rgb_val, 256),
        )).astype(np.uint8)

        all_resized_objs[i] = add_color_to_obj(
            tmp_obj,
            tmp_base_rgb,
            grad_color_ratio_between_end_points,
            n_grad_color,
            light_color_on_top,
        )

    final_img = Image.fromarray(np.zeros((tex_h, tex_w, 4), dtype=np.uint8))

    # gen plane image from back to front
    plane_imgs = []
    for plane_idx in range(n_planes - 1, -1, -1):

        tmp_plane_img = Image.fromarray(
            np.zeros((tex_h, tex_w, 4), dtype=np.uint8))

        obj_idxs = np.where(obj_plane_idxs == plane_idx)[0]

        for tmp_idx in obj_idxs:

            tmp_obj = all_resized_objs[tmp_idx]
            min_row, max_row, min_col, max_col, _ = obj_rects[tmp_idx]
            tmp_img = np.zeros((tex_h, tex_w, 4), dtype=np.uint8)
            tmp_img[min_row:max_row, min_col:max_col, :] = tmp_obj
            tmp_img = Image.fromarray(tmp_img)

            # alpha-composition
            tmp_plane_img = Image.alpha_composite(tmp_plane_img, tmp_img)

        plane_imgs.append(np.array(tmp_plane_img))

    # NOTE: 1st elem in the furthest plane
    return plane_imgs


def compute_intersection_between_cam_frustum_and_plane(cam_pos, ray_dir,
                                                       z_plane):
    """We currently assume the plane is perpendicular to Z axis of world
    coordinate system.

    Args:
        cam_pos ([type]): [description]
        ray_dir ([type]): [description]
        plane_z ([type]): [description]
    """
    n, _, h, w = ray_dir.shape

    # find step size of rays
    z_cam = cam_pos[:, 2:3]
    z_ray = ray_dir[:, 2:3]
    z_diff = (z_plane - z_cam).view(n, 1, 1, 1)
    z_diff = z_diff.expand(n, 1, h, w)
    scale = z_diff / z_ray

    # compute ray-plane intersections
    xyz = cam_pos.view(-1, 3, 1, 1) + ray_dir * scale
    x, y = xyz[:, 0, :, :], xyz[:, 1, :, :]

    bound_dict = {
        'min_x': torch.min(x).cpu().numpy(),
        'max_x': torch.max(x).cpu().numpy(),
        'min_y': torch.min(y).cpu().numpy(),
        'max_y': torch.max(y).cpu().numpy(),
    }

    return bound_dict


def compute_plane_dhws_given_cam_pose_spatial_range(
        camera: Camera,
        sphere_center: np.ndarray,
        sphere_r: Union[float, None],
        cam_horizontal_min: float,
        cam_horizontal_max: float,
        cam_vertical_min: float,
        cam_vertical_max: float,
        cam_pose_n_truncated_stds: int,
        plane_zs: List[float],
        enlarge_factor: float = 1.0,
        device: torch.device = torch.device('cpu'),
):
    """This function computes planes's spatial height and width of MPI given
    camera's pose range.

    Args:
        camera (Camera): [description]
        sphere_center (np.ndarray): [description]
        sphere_r (Union[float, None]): [description]
        cam_horizontal_min (float): [description]
        cam_horizontal_max (float): [description]
        cam_vertical_min (float): [description]
        cam_vertical_max (float): [description]
        plane_zs (np.ndarray): [description]
        enlarge_factor (float, optional): [description]. Defaults to 1.25.
        device (torch.device, optional): [description]. Defaults to torch.device("cpu").

    Returns:
        [type]: [description]
    """

    # cam_horizontal_angles = [cam_horizontal_min, cam_horizontal_max]
    # cam_vertical_angles = [cam_vertical_min, cam_vertical_max]

    horizontal_mid_angle = (cam_horizontal_min + cam_horizontal_max) / 2
    vertical_mid_angle = (cam_vertical_min + cam_vertical_max) / 2

    cam_heuristic_angles = []
    _N = 100
    horizontal_angle_list = np.linspace(cam_horizontal_min, cam_horizontal_max,
                                        _N)
    vertical_angle_list = np.linspace(cam_vertical_min, cam_vertical_max, _N)
    for tmp_h in horizontal_angle_list:
        for tmp_v in vertical_angle_list:
            cam_heuristic_angles.append((tmp_h, tmp_v))

    # NOTE: we need this to compute the base_spatial_size
    cam_heuristic_angles.append((horizontal_mid_angle, vertical_mid_angle))

    # print("\ncam_heuristic_angles: ", cam_heuristic_angles, "\n")
    print('\n', sphere_center, sphere_r, cam_pose_n_truncated_stds, '\n')

    plane_bound_val = {'min_x': [], 'max_x': [], 'min_y': [], 'max_y': []}

    for i in tqdm.tqdm(range(len(cam_heuristic_angles))):
        # we set std to 0 so that we just choose angle to be the mean
        cur_c2w_mats, _, _ = gen_sphere_path(
            n_cams=1,
            sphere_center=sphere_center,
            sphere_r=sphere_r,
            yaw_mean=cam_heuristic_angles[i][0],
            yaw_std=0,
            pitch_mean=cam_heuristic_angles[i][1],
            pitch_std=0,
            sample_method='uniform',
            n_truncated_stds=cam_pose_n_truncated_stds,
            device=device,
        )

        ray_dir, cam_pos, z_dir = camera.generate_rays(
            cur_c2w_mats[0, ...], border_only=True)
        ray_dir = torch.from_numpy(ray_dir).unsqueeze(0).float().to(device)
        cam_pos = torch.from_numpy(cam_pos).view(1, 3).float().to(device)

        # NOTE: we assume depths are stored from front to back
        cur_bound_val = compute_intersection_between_cam_frustum_and_plane(
            cam_pos, ray_dir, plane_zs[-1])
        # print("\n", cam_heuristic_angles[i], cur_bound_val, "\n")
        for k in plane_bound_val:
            plane_bound_val[k].append(cur_bound_val[k])

        if i == len(cam_heuristic_angles) - 1:
            # NOTE: this is for (horizontal_mid_angle, vertical_mid_angle).
            # We need this spatial resolution as the BASE resolution.
            # We assume square plane.
            base_spatial_size = min(
                plane_bound_val['max_x'][-1] - plane_bound_val['min_x'][-1],
                plane_bound_val['max_y'][-1] - plane_bound_val['min_y'][-1],
            )
            # we use this spatial size as the confined size
            confined_spatial_h = 2 * np.max([
                np.abs(plane_bound_val['min_y'][-1]),
                np.abs(plane_bound_val['max_y'][-1])
            ])
            confined_spatial_w = 2 * np.max([
                np.abs(plane_bound_val['min_x'][-1]),
                np.abs(plane_bound_val['max_x'][-1])
            ])

    plane_bound_val['min_x'] = np.min(plane_bound_val['min_x'])
    plane_bound_val['max_x'] = np.max(plane_bound_val['max_x'])
    plane_bound_val['min_y'] = np.min(plane_bound_val['min_y'])
    plane_bound_val['max_y'] = np.max(plane_bound_val['max_y'])

    print('\nplane_bound_val: ', plane_bound_val, '\n')

    # NOTE: 5.0 is a heuristic value, indicating the plane is too large.
    plane_bound_max_val = np.max(np.abs(list(plane_bound_val.values())))
    assert plane_bound_max_val <= 5.0, (
        f"You have MPI's plane whose boundary value is up to {plane_bound_max_val}. "
        f"This usually means the camera poses's range is too big, which will cause problems for MPI representation. "
        f'Please reduce h_stddev or v_stddev in curriculums.py or cam_pose_n_truncated_stds in config file.'
    )

    # for simplicity, we enforce plane to be symmetric
    # Meanhile, we have +X right, +Y down.
    spatial_h = 2 * np.max(
        [np.abs(plane_bound_val['min_y']),
         np.abs(plane_bound_val['max_y'])])
    spatial_w = 2 * np.max(
        [np.abs(plane_bound_val['min_x']),
         np.abs(plane_bound_val['max_x'])])

    spatial_h = spatial_h * enlarge_factor
    spatial_w = spatial_w * enlarge_factor

    dhws = [[plane_zs[-1], spatial_h, spatial_w]]
    for i in range(len(plane_zs) - 2, -1, -1):
        tmp_z = plane_zs[i]
        tmp_spatial_h = confined_spatial_h * tmp_z / plane_zs[-1]
        tmp_spatial_w = confined_spatial_w * tmp_z / plane_zs[-1]
        dhws.append([tmp_z, tmp_spatial_h, tmp_spatial_w])

    # we make dhws store values with the order from 1st plane to last one
    dhws.reverse()
    dhws = np.array(dhws)

    # NOTE: We need to think about this later.
    tex_expand_ratio = np.max(dhws[:, 1:] / base_spatial_size)
    # tex_expand_ratio = enlarge_factor

    return dhws, tex_expand_ratio


def compute_plane_dhws_given_cam_pose_spatial_range_confined(
        camera: Camera,
        sphere_center: np.ndarray,
        sphere_r: Union[float, None],
        cam_horizontal_min: float,
        cam_horizontal_max: float,
        cam_vertical_min: float,
        cam_vertical_max: float,
        cam_pose_n_truncated_stds: int,
        plane_zs: List[float],
        enlarge_factor: float = 1.0,
        device: torch.device = torch.device('cpu'),
):
    """This function computes planes's spatial height and width of MPI given
    camera's pose range.

    Args:
        camera (Camera): [description]
        sphere_center (np.ndarray): [description]
        sphere_r (Union[float, None]): [description]
        cam_horizontal_min (float): [description]
        cam_horizontal_max (float): [description]
        cam_vertical_min (float): [description]
        cam_vertical_max (float): [description]
        plane_zs (np.ndarray): [description]
        enlarge_factor (float, optional): [description]. Defaults to 1.25.
        device (torch.device, optional): [description]. Defaults to torch.device("cpu").

    Returns:
        [type]: [description]
    """

    # cam_horizontal_angles = [cam_horizontal_min, cam_horizontal_max]
    # cam_vertical_angles = [cam_vertical_min, cam_vertical_max]

    horizontal_mid_angle = (cam_horizontal_min + cam_horizontal_max) / 2
    vertical_mid_angle = (cam_vertical_min + cam_vertical_max) / 2

    cam_heuristic_angles = []
    _N = 100
    horizontal_angle_list = np.linspace(cam_horizontal_min, cam_horizontal_max,
                                        _N)
    vertical_angle_list = np.linspace(cam_vertical_min, cam_vertical_max, _N)
    for tmp_h in horizontal_angle_list:
        for tmp_v in vertical_angle_list:
            cam_heuristic_angles.append((tmp_h, tmp_v))

    # NOTE: we need this to compute the base_spatial_size
    cam_heuristic_angles.append((horizontal_mid_angle, vertical_mid_angle))

    # print("\ncam_heuristic_angles: ", cam_heuristic_angles, "\n")
    print('\n', sphere_center, sphere_r, cam_pose_n_truncated_stds, '\n')

    plane_bound_val = {'min_x': [], 'max_x': [], 'min_y': [], 'max_y': []}

    for i in tqdm.tqdm(range(len(cam_heuristic_angles))):
        # we set std to 0 so that we just choose angle to be the mean
        cur_c2w_mats, _, _ = gen_sphere_path(
            n_cams=1,
            sphere_center=sphere_center,
            sphere_r=sphere_r,
            yaw_mean=cam_heuristic_angles[i][0],
            yaw_std=0,
            pitch_mean=cam_heuristic_angles[i][1],
            pitch_std=0,
            sample_method='uniform',
            n_truncated_stds=cam_pose_n_truncated_stds,
            device=device,
        )

        ray_dir, cam_pos, z_dir = camera.generate_rays(
            cur_c2w_mats[0, ...], border_only=True)
        ray_dir = torch.from_numpy(ray_dir).unsqueeze(0).float().to(device)
        cam_pos = torch.from_numpy(cam_pos).view(1, 3).float().to(device)

        # NOTE: we assume depths are stored from front to back
        cur_bound_val = compute_intersection_between_cam_frustum_and_plane(
            cam_pos, ray_dir, plane_zs[-1])
        # print("\n", cam_heuristic_angles[i], cam_pos[:, 2:], cur_bound_val, "\n")
        for k in plane_bound_val:
            plane_bound_val[k].append(cur_bound_val[k])

        if i == len(cam_heuristic_angles) - 1:
            # NOTE: this is for (horizontal_mid_angle, vertical_mid_angle).
            # We need this spatial resolution as the BASE resolution.
            # We assume square plane.
            base_spatial_size = min(
                plane_bound_val['max_x'][-1] - plane_bound_val['min_x'][-1],
                plane_bound_val['max_y'][-1] - plane_bound_val['min_y'][-1],
            )
            # we use this spatial size as the confined size
            confined_spatial_h = 2 * np.max([
                np.abs(plane_bound_val['min_y'][-1]),
                np.abs(plane_bound_val['max_y'][-1])
            ])
            confined_spatial_w = 2 * np.max([
                np.abs(plane_bound_val['min_x'][-1]),
                np.abs(plane_bound_val['max_x'][-1])
            ])

    plane_bound_val['min_x'] = np.min(plane_bound_val['min_x'])
    plane_bound_val['max_x'] = np.max(plane_bound_val['max_x'])
    plane_bound_val['min_y'] = np.min(plane_bound_val['min_y'])
    plane_bound_val['max_y'] = np.max(plane_bound_val['max_y'])

    print('\nplane_bound_val: ', plane_bound_val, '\n')

    # NOTE: 5.0 is a heuristic value, indicating the plane is too large.
    plane_bound_max_val = np.max(np.abs(list(plane_bound_val.values())))
    assert plane_bound_max_val <= 5.0, (
        f"You have MPI's plane whose boundary value is up to {plane_bound_max_val}. "
        f"This usually means the camera poses's range is too big, which will cause problems for MPI representation. "
        f'Please reduce h_stddev or v_stddev in curriculums.py or cam_pose_n_truncated_stds in config file.'
    )

    # for simplicity, we enforce plane to be symmetric
    # Meanhile, we have +X right, +Y down.
    spatial_h = 2 * np.max(
        [np.abs(plane_bound_val['min_y']),
         np.abs(plane_bound_val['max_y'])])
    spatial_w = 2 * np.max(
        [np.abs(plane_bound_val['min_x']),
         np.abs(plane_bound_val['max_x'])])

    spatial_h = spatial_h * enlarge_factor
    spatial_w = spatial_w * enlarge_factor

    dhws = [[plane_zs[-1], spatial_h, spatial_w]]
    for i in range(len(plane_zs) - 2, -1, -1):
        tmp_z = plane_zs[i]
        dhws.append([tmp_z, confined_spatial_h, confined_spatial_w])

    # we make dhws store values with the order from 1st plane to last one
    dhws.reverse()
    dhws = np.array(dhws)

    # NOTE: We need to think about this later.
    tex_expand_ratio = np.max(dhws[:, 1:] / base_spatial_size)
    # tex_expand_ratio = enlarge_factor

    return dhws, tex_expand_ratio
