# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from mmengine.dist import master_only
from mmengine.visualization import Visualizer as BaseVisualizer
from torch import Tensor
from torchvision.utils import make_grid

from mmagic.registry import VISUALIZERS
from mmagic.structures import DataSample
from mmagic.utils.typing import SampleList

mean_std_type = Optional[Sequence[Union[float, int]]]


@VISUALIZERS.register_module()
class Visualizer(BaseVisualizer):
    """MMagic Visualizer.

    Args:
        name (str): Name of the instance. Defaults to 'visualizer'.
        vis_backends (list, optional): Visual backend config list.
            Defaults to None.
        save_dir (str, optional): Save file dir for all storage backends.
            If it is None, the backend storage will not save any data.

    Examples::

        >>> # Draw image
        >>> vis = Visualizer()
        >>> vis.add_datasample(
        >>>     'random_noise',
        >>>     gen_samples=torch.rand(2, 3, 10, 10),
        >>>     gt_samples=dict(imgs=torch.randn(2, 3, 10, 10)),
        >>>     gt_keys='imgs',
        >>>     vis_mode='image',
        >>>     n_rows=2,
        >>>     step=10)
    """

    def __init__(self,
                 name='visualizer',
                 vis_backends: Optional[List[Dict]] = None,
                 save_dir: Optional[str] = None) -> None:
        super().__init__(name, vis_backends=vis_backends, save_dir=save_dir)

    @staticmethod
    def _post_process_image(image: Tensor) -> Tensor:
        """Post process images.

        Args:
            image (Tensor): Image to post process. The value range of
                image should be in [0, 255] and the channel order should
                be BGR.

        Returns:
            Tensor: Image in RGB color order.
        """
        # input image should be [0, 255] and BGR
        if image.shape[1] == 1:
            image = torch.cat([image, image, image], dim=1)
        image = image[:, [2, 1, 0], ...]
        return image

    @staticmethod
    def _get_n_row_and_padding(samples: Tuple[dict, Tensor],
                               n_row: Optional[int] = None
                               ) -> Tuple[int, Optional[Tensor]]:
        """Get number of sample in each row and tensor for padding the empty
        position.

        Args:
            samples (Tuple[dict, Tensor]): Samples to visualize.
            n_row (int, optional): Number of images displayed in each row of.
                If not passed, n_row will be set as ``int(sqrt(batch_size))``.

        Returns:
            Tuple[int, Optional[int]]: Number of sample in each row and tensor
                for padding the empty position.
        """
        if isinstance(samples, dict):
            for sample in iter(samples.values()):
                # NOTE: dirty way to get the shape of image tensor
                if isinstance(sample, Tensor) and sample.ndim in [4, 5]:
                    sample_shape = sample.shape
                    break
        else:
            sample_shape = samples.shape

        n_samples = sample_shape[0]
        if n_row is None:
            n_row = math.ceil(math.sqrt(n_samples))
        if n_samples % n_row == 0:
            n_padding = 0
        else:
            n_padding = n_row - (n_samples % n_row)
        if n_padding:
            return n_row, -1.0 * torch.ones(n_padding, *sample_shape[1:])
        return n_row, None

    def _vis_gif_sample(self, gen_samples: SampleList,
                        target_keys: Union[str, List[str],
                                           None], n_row: int) -> np.ndarray:
        """Visualize gif samples.

        Args:
            gen_samples (SampleList): List of data samples to visualize
            target_keys (Union[str, List[str], None]): Keys of the
                visualization target in data samples.
            n_rows (int, optional): Number of images in one row.

        Returns:
            np.ndarray: The visualization results.
        """

        def post_process_sequence(samples):
            num_timesteps = samples.shape[1]
            seq_list = [
                self._post_process_image(samples[:, t, ...].cpu())
                for t in range(num_timesteps)
            ]
            return torch.stack(seq_list, dim=1)

        if target_keys is None:
            # get all the keys that are tensors with 4 dimensions
            target_keys = [
                k for k, v in gen_samples[0].items()
                if ((not k.startswith('_')) and (isinstance(v, Tensor)) and (
                    v.data.ndim == 4))
            ]
        target_keys = [target_keys] if isinstance(target_keys, str) \
            else target_keys

        sample_list = list()
        for sample in gen_samples:
            sample_dict = dict()
            for k in target_keys:
                sample_dict[k] = self._get_vis_data_by_key(sample, k)
            sample_list.append(sample_dict)

        for k in sample_list[0].keys():
            sample_ = torch.stack([samp[k] for samp in sample_list], dim=0)
            sample_ = post_process_sequence(sample_.cpu())
            sample_dict[k] = sample_

        n_row, padding_tensor = self._get_n_row_and_padding(sample_dict, n_row)
        num_timesteps = next(iter(sample_dict.values())).shape[1]

        vis_results = []
        for sample in sample_dict.values():
            if padding_tensor is not None:
                vis_results.append(torch.cat([sample, padding_tensor], dim=0))
            else:
                vis_results.append(sample)

        # concatenate along batch size
        vis_results = torch.cat(vis_results)
        vis_results = [
            make_grid(vis_results[:, t, ...].cpu(),
                      nrow=n_row).cpu().permute(1, 2, 0)
            for t in range(num_timesteps)
        ]
        vis_results = torch.stack(vis_results, dim=0)  # [t, H, W, 3]
        vis_results = vis_results.numpy().astype(np.uint8)
        return vis_results

    def _vis_image_sample(self, gen_samples: SampleList,
                          target_keys: Union[str, List[str],
                                             None], n_row: int) -> np.ndarray:
        """Visualize image samples.

        Args:
            gen_samples (SampleList): List of data samples to visualize
            target_keys (Union[str, List[str], None]): Keys of the
                visualization target in data samples.
            color_order (str): The color order of the passed images.
            target_mean (Sequence[Union[float, int]]): The target mean of the
                visualization results.
            target_std (Sequence[Union[float, int]]): The target std of the
                visualization results.
            n_rows (int, optional): Number of images in one row.

        Returns:
            np.ndarray: The visualization results.
        """
        if target_keys is None:
            # get all key of image tensor automatically
            target_keys = [
                k for k, v in gen_samples[0].items()
                if ((not k.startswith('_')) and (isinstance(v, Tensor)) and (
                    v.data.ndim == 3))
            ]
        target_keys = [target_keys] if isinstance(target_keys, str) \
            else target_keys

        sample_list = list()
        for sample in gen_samples:
            sample_dict = dict()
            for k in target_keys:
                sample_dict[k] = self._get_vis_data_by_key(sample, k)
            sample_list.append(sample_dict)

        for k in sample_list[0].keys():
            sample_ = torch.stack([samp[k] for samp in sample_list], dim=0)
            sample_ = self._post_process_image(sample_.cpu())
            sample_dict[k] = sample_

        n_row, padding_tensor = self._get_n_row_and_padding(sample_dict, n_row)

        vis_results = []
        for sample in sample_dict.values():
            if padding_tensor is not None:
                vis_results.append(torch.cat([sample, padding_tensor], dim=0))
            else:
                vis_results.append(sample)

        # concatenate along batch size
        vis_results = torch.cat(vis_results, dim=0)
        vis_results = make_grid(vis_results, nrow=n_row).cpu().permute(1, 2, 0)
        vis_results = vis_results.numpy().astype(np.uint8)
        return vis_results

    def _get_vis_data_by_key(self, sample: DataSample, key: str) -> Tensor:
        """Get tensor in ``DataSample`` by the given key.

        Args:
            sample (DataSample): Input data sample.
            key (str): Name of the target tensor.

        Returns:
            Tensor: Tensor from the data sample.
        """
        if '.' in key:
            key_list = key.split('.')
        else:
            key_list = [key]
        vis_data = sample
        for k in key_list:
            # get vis data step by step
            assert hasattr(vis_data, k)
            vis_data = getattr(vis_data, k)

        if isinstance(vis_data, Tensor):
            return vis_data
        else:
            # check only one tensor in current datasample to visualize
            elements = [
                element for k, element in vis_data.items()
                if not k.startswith('_') and isinstance(element, Tensor)
            ]
            assert len(elements) == 1, (
                f'Find {len(elements)} Tensor in DataSample with '
                f'key {key}.')
            vis_data = elements[0]

            assert isinstance(
                vis_data,
                Tensor), (f'Element with key \'{key}\' is not a Tensor.')
            return vis_data

    def add_datasample(self,
                       name: str,
                       *,
                       gen_samples: Sequence[DataSample],
                       target_keys: Optional[Tuple[str, List[str]]] = None,
                       vis_mode: Optional[str] = None,
                       n_row: Optional[int] = None,
                       show: bool = False,
                       wait_time: int = 0,
                       step: int = 0,
                       **kwargs) -> None:
        """Draw datasample and save to all backends.

        If GT and prediction are plotted at the same time, they
        are displayed in a stitched image where the left image is the
        ground truth and the right image is the prediction.

        If ``show`` is True, all storage backends are ignored,
        and the images will be displayed in a local window.

        Args:
            name (str): The image identifier.
            gen_samples (List[DataSample]): Data samples to visualize.
            vis_mode (str, optional): Visualization mode. If not passed, will
                visualize results as image. Defaults to None.
            n_rows (int, optional): Number of images in one row.
                Defaults to None.
            color_order (str): The color order of the passed images. Defaults
                to 'bgr'.
            target_mean (Sequence[Union[float, int]]): The target mean of the
                visualization results. Defaults to 127.5.
            target_std (Sequence[Union[float, int]]): The target std of the
                visualization results. Defaults to 127.5.
            show (bool): Whether to display the drawn image. Default to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            step (int): Global step value to record. Defaults to 0.
        """

        # get visualize function
        if vis_mode is None:
            vis_func = self._vis_image_sample
        else:
            vis_func = getattr(self, f'_vis_{vis_mode}_sample')

        vis_sample = vis_func(gen_samples, target_keys, n_row)

        if show:
            self.show(vis_sample, win_name=name, wait_time=wait_time)

        self.add_image(name, vis_sample, step, **kwargs)

    @master_only
    def add_image(self,
                  name: str,
                  image: np.ndarray,
                  step: int = 0,
                  **kwargs) -> None:
        """Record the image. Support input kwargs.

        Args:
            name (str): The image identifier.
            image (np.ndarray, optional): The image to be saved. The format
                should be RGB. Default to None.
            step (int): Global step value to record. Default to 0.
        """
        for vis_backend in self._vis_backends.values():
            vis_backend.add_image(name, image, step, **kwargs)  # type: ignore
