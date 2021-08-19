# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

from ..registry import PIPELINES
from .utils import make_coord


@PIPELINES.register_module()
class GenerateHeatmap:
    """Generate heatmap from keypoint.

    Args:
        keypoint (str): Key of keypoint in dict.
        ori_size (int | Tuple[int]): Original image size of keypoint.
        target_size (int | Tuple[int]): Target size of heatmap.
        sigma (float): Sigma parameter of heatmap. Default: 1.0
    """

    def __init__(self, keypoint, ori_size, target_size, sigma=1.0):
        if isinstance(ori_size, int):
            ori_size = (ori_size, ori_size)
        else:
            ori_size = ori_size[:2]
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        else:
            target_size = target_size[:2]
        self.size_ratio = (target_size[0] / ori_size[0],
                           target_size[1] / ori_size[1])
        self.keypoint = keypoint
        self.sigma = sigma
        self.target_size = target_size
        self.ori_size = ori_size

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation. Require keypoint.

        Returns:
            dict: A dict containing the processed data and information.
                Add 'heatmap'.
        """
        keypoint_list = [(keypoint[0] * self.size_ratio[0],
                          keypoint[1] * self.size_ratio[1])
                         for keypoint in results[self.keypoint]]
        heatmap_list = [
            self._generate_one_heatmap(keypoint) for keypoint in keypoint_list
        ]
        results['heatmap'] = np.stack(heatmap_list, axis=2)
        return results

    def _generate_one_heatmap(self, keypoint):
        """Generate One Heatmap.

        Args:
            landmark (Tuple[float]): Location of a landmark.

        results:
            heatmap (np.ndarray): A heatmap of landmark.
        """
        w, h = self.target_size

        x_range = np.arange(start=0, stop=w, dtype=int)
        y_range = np.arange(start=0, stop=h, dtype=int)
        grid_x, grid_y = np.meshgrid(x_range, y_range)
        dist2 = (grid_x - keypoint[0])**2 + (grid_y - keypoint[1])**2
        exponent = dist2 / 2.0 / self.sigma / self.sigma
        heatmap = np.exp(-exponent)
        return heatmap

    def __repr__(self):
        return (f'{self.__class__.__name__}, '
                f'keypoint={self.keypoint}, '
                f'ori_size={self.ori_size}, '
                f'target_size={self.target_size}, '
                f'sigma={self.sigma}')


@PIPELINES.register_module()
class GenerateCoordinateAndCell:
    """Generate coordinate and cell.

    Generate coordinate from the desired size of SR image.
        Train or val:
            1. Generate coordinate from GT.
            2. Reshape GT image to (HgWg, 3) and transpose to (3, HgWg).
                where `Hg` and `Wg` represent the height and width of GT.
        Test:
            Generate coordinate from LQ and scale or target_size.
    Then generate cell from coordinate.

    Args:
        sample_quantity (int): The quantity of samples in coordinates.
            To ensure that the GT tensors in a batch have the same dimensions.
            Default: None.
        scale (float): Scale of upsampling.
            Default: None.
        target_size (tuple[int]): Size of target image.
            Default: None.

    The priority of getting 'size of target image' is:
        1, results['gt'].shape[-2:]
        2, results['lq'].shape[-2:] * scale
        3, target_size
    """

    def __init__(self, sample_quantity=None, scale=None, target_size=None):
        self.sample_quantity = sample_quantity
        self.scale = scale
        self.target_size = target_size

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.
                Require either in results:
                    1. 'lq' (tensor), whose shape is similar as (3, H, W).
                    2. 'gt' (tensor), whose shape is similar as (3, H, W).
                    3. None, the premise is
                        self.target_size and len(self.target_size) >= 2.

        Returns:
            dict: A dict containing the processed data and information.
                Reshape 'gt' to (-1, 3) and transpose to (3, -1) if 'gt'
                in results.
                Add 'coord' and 'cell'.
        """
        # generate hr_coord (and hr_rgb)
        if 'gt' in results:
            crop_hr = results['gt']
            self.target_size = crop_hr.shape
            hr_rgb = crop_hr.contiguous().view(3, -1).permute(1, 0)
            results['gt'] = hr_rgb
        elif self.scale is not None and 'lq' in results:
            _, h_lr, w_lr = results['lq'].shape
            self.target_size = (round(h_lr * self.scale),
                                round(w_lr * self.scale))
        else:
            assert self.target_size is not None
            assert len(self.target_size) >= 2
        hr_coord = make_coord(self.target_size[-2:])

        if self.sample_quantity is not None and 'gt' in results:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_quantity, replace=False)
            hr_coord = hr_coord[sample_lst]
            results['gt'] = results['gt'][sample_lst]

        # Preparations for cell decoding
        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / self.target_size[-2]
        cell[:, 1] *= 2 / self.target_size[-1]

        results['coord'] = hr_coord
        results['cell'] = cell

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'sample_quantity={self.sample_quantity}, '
                     f'scale={self.scale}, target_size={self.target_size}')
        return repr_str
