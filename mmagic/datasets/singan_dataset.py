# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import mmcv
import numpy as np
from mmengine.dataset import BaseDataset

from mmagic.registry import DATASETS


def create_real_pyramid(real, min_size, max_size, scale_factor_init):
    """Create image pyramid.

    This function is modified from the official implementation:
    https://github.com/tamarott/SinGAN/blob/master/SinGAN/functions.py#L221

    In this implementation, we adopt the rescaling function from MMCV.
    Args:
        real (np.array): The real image array.
        min_size (int): The minimum size for the image pyramid.
        max_size (int): The maximum size for the image pyramid.
        scale_factor_init (float): The initial scale factor.
    """

    num_scales = int(
        np.ceil(
            np.log(np.power(min_size / min(real.shape[0], real.shape[1]), 1)) /
            np.log(scale_factor_init))) + 1

    scale2stop = int(
        np.ceil(
            np.log(
                min([max_size, max([real.shape[0], real.shape[1]])]) /
                max([real.shape[0], real.shape[1]])) /
            np.log(scale_factor_init)))

    stop_scale = num_scales - scale2stop

    scale1 = min(max_size / max([real.shape[0], real.shape[1]]), 1)
    real_max = mmcv.imrescale(real, scale1)
    scale_factor = np.power(
        min_size / (min(real_max.shape[0], real_max.shape[1])),
        1 / (stop_scale))

    scale2stop = int(
        np.ceil(
            np.log(
                min([max_size, max([real.shape[0], real.shape[1]])]) /
                max([real.shape[0], real.shape[1]])) /
            np.log(scale_factor_init)))
    stop_scale = num_scales - scale2stop

    reals = []
    for i in range(stop_scale + 1):
        scale = np.power(scale_factor, stop_scale - i)
        curr_real = mmcv.imrescale(real, scale)
        reals.append(curr_real)

    return reals, scale_factor, stop_scale


@DATASETS.register_module()
class SinGANDataset(BaseDataset):
    """SinGAN Dataset.

    In this dataset, we create an image pyramid and save it in the cache.

    Args:
        img_path (str): Path to the single image file.
        min_size (int): Min size of the image pyramid. Here, the number will be
            set to the ``min(H, W)``.
        max_size (int): Max size of the image pyramid. Here, the number will be
            set to the ``max(H, W)``.
        scale_factor_init (float): Rescale factor. Note that the actual factor
            we use may be a little bit different from this value.
        num_samples (int, optional): The number of samples (length) in this
            dataset. Defaults to -1.
    """

    def __init__(self,
                 data_root,
                 min_size,
                 max_size,
                 scale_factor_init,
                 pipeline,
                 num_samples=-1):
        self.min_size = min_size
        self.max_size = max_size
        self.scale_factor_init = scale_factor_init
        self.num_samples = num_samples
        super().__init__(data_root=data_root, pipeline=pipeline)

    def full_init(self):
        """Skip the full init process for SinGANDataset."""

        self.load_data_list(self.min_size, self.max_size,
                            self.scale_factor_init)

    def load_data_list(self, min_size, max_size, scale_factor_init):
        """Load annotations for SinGAN Dataset.

        Args:
            min_size (int): The minimum size for the image pyramid.
            max_size (int): The maximum size for the image pyramid.
            scale_factor_init (float): The initial scale factor.
        """
        real = mmcv.imread(self.data_root)
        self.reals, self.scale_factor, self.stop_scale = create_real_pyramid(
            real, min_size, max_size, scale_factor_init)

        self.data_dict = {}

        for i, real in enumerate(self.reals):
            self.data_dict[f'real_scale{i}'] = real

        self.data_dict['input_sample'] = np.zeros_like(
            self.data_dict['real_scale0']).astype(np.float32)

    def __getitem__(self, index):
        """Get `:attr:self.data_dict`. For SinGAN, we use single image with
        different resolution to train the model.

        Args:
            idx (int): This will be ignored in :class:`SinGANDataset`.

        Returns:
            dict: Dict contains input image in different resolution.
            ``self.pipeline``.
        """
        return self.pipeline(deepcopy(self.data_dict))

    def __len__(self):
        """Get the length of filtered dataset and automatically call
        ``full_init`` if the  dataset has not been fully init.

        Returns:
            int: The length of filtered dataset.
        """
        return int(1e6) if self.num_samples < 0 else self.num_samples
