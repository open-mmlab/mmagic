# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.dist import all_gather, get_world_size

from mmagic.registry import METRICS
from .base_gen_metric import GenMetric


def sliced_wasserstein(distribution_a,
                       distribution_b,
                       dir_repeats=4,
                       dirs_per_repeat=128):
    r"""sliced Wasserstein distance of two sets of patches.

    Ref: https://github.com/tkarras/progressive_growing_of_gans/blob/master/metrics/ms_ssim.py  # noqa

    Args:
        distribution_a (Tensor): Descriptors of first distribution.
        distribution_b (Tensor): Descriptors of second distribution.
        dir_repeats (int): The number of projection times. Default to 4.
        dirs_per_repeat (int): The number of directions per projection.
            Default to 128.

    Returns:
        float: sliced Wasserstein distance.
    """
    if torch.cuda.is_available():
        distribution_b = distribution_b.cuda()
    assert distribution_a.ndim == 2
    assert distribution_a.shape == distribution_b.shape
    assert dir_repeats > 0 and dirs_per_repeat > 0
    distribution_a = distribution_a.to(distribution_b.device)
    results = []
    for _ in range(dir_repeats):
        dirs = torch.randn(distribution_a.shape[1], dirs_per_repeat)
        dirs /= torch.sqrt(torch.sum((dirs**2), dim=0, keepdim=True))
        dirs = dirs.to(distribution_b.device)
        proj_a = torch.matmul(distribution_a, dirs)
        proj_b = torch.matmul(distribution_b, dirs)
        # To save cuda memory, we perform sort in cpu
        proj_a, _ = torch.sort(proj_a.cpu(), dim=0)
        proj_b, _ = torch.sort(proj_b.cpu(), dim=0)
        dists = torch.abs(proj_a - proj_b)
        results.append(torch.mean(dists).item())
    torch.cuda.empty_cache()
    return sum(results) / dir_repeats


# Gaussian blur kernel
def get_gaussian_kernel():
    """Get the gaussian blur kernel.

    Returns:
        Tensor: Blur kernel.
    """
    kernel = np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6],
                       [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]],
                      np.float32) / 256.0
    gaussian_k = torch.as_tensor(kernel.reshape(1, 1, 5, 5))
    return gaussian_k


def get_pyramid_layer(image, gaussian_k, direction='down'):
    """Get the pyramid layer.

    Args:
        image (Tensor): Input image.
        gaussian_k (Tensor): Gaussian kernel
        direction (str, optional): The direction of pyramid. Defaults to
            'down'.

    Returns:
        Tensor: The output of the pyramid.
    """
    gaussian_k = gaussian_k.to(image.device)
    if direction == 'up':
        image = F.interpolate(image, scale_factor=2)
    multiband = [
        F.conv2d(
            image[:, i:i + 1, :, :],
            gaussian_k,
            padding=2,
            stride=1 if direction == 'up' else 2) for i in range(3)
    ]
    image = torch.cat(multiband, dim=1)
    return image


def gaussian_pyramid(original, n_pyramids, gaussian_k):
    """Get a group of gaussian pyramid.

    Args:
        original (Tensor): The input image.
        n_pyramids (int): The number of pyramids.
        gaussian_k (Tensor): The gaussian kernel.

    Returns:
        List[Tensor]: The list of output of gaussian pyramid.
    """
    x = original
    # pyramid down
    pyramids = [original]
    for _ in range(n_pyramids):
        x = get_pyramid_layer(x, gaussian_k)
        pyramids.append(x)
    return pyramids


def laplacian_pyramid(original, n_pyramids, gaussian_k):
    """Calculate Laplacian pyramid.

    Ref: https://github.com/koshian2/swd-pytorch/blob/master/swd.py

    Args:
        original (Tensor): Batch of Images with range [0, 1] and order "NCHW"
        n_pyramids (int): Levels of pyramids minus one.
        gaussian_k (Tensor): Gaussian kernel with shape (1, 1, 5, 5).

    Return:
        list[Tensor]. Laplacian pyramids of original.
    """
    # create gaussian pyramid
    pyramids = gaussian_pyramid(original, n_pyramids, gaussian_k)

    # pyramid up - diff
    laplacian = []
    for i in range(len(pyramids) - 1):
        diff = pyramids[i] - get_pyramid_layer(pyramids[i + 1], gaussian_k,
                                               'up')
        laplacian.append(diff)
    # Add last gaussian pyramid
    laplacian.append(pyramids[len(pyramids) - 1])
    return laplacian


def get_descriptors_for_minibatch(minibatch, nhood_size, nhoods_per_image):
    r"""Get descriptors of one level of pyramids.

    Ref: https://github.com/tkarras/progressive_growing_of_gans/blob/master/metrics/sliced_wasserstein.py  # noqa

    Args:
        minibatch (Tensor): Pyramids of one level with order "NCHW".
        nhood_size (int): Pixel neighborhood size.
        nhoods_per_image (int): The number of descriptors per image.

    Return:
        Tensor: Descriptors of images from one level batch.
    """
    S = minibatch.shape  # (minibatch, channel, height, width)
    assert len(S) == 4 and S[1] == 3
    N = nhoods_per_image * S[0]
    H = nhood_size // 2
    nhood, chan, x, y = np.ogrid[0:N, 0:3, -H:H + 1, -H:H + 1]
    img = nhood // nhoods_per_image
    x = x + np.random.randint(H, S[3] - H, size=(N, 1, 1, 1))
    y = y + np.random.randint(H, S[2] - H, size=(N, 1, 1, 1))
    idx = ((img * S[1] + chan) * S[2] + y) * S[3] + x
    return minibatch.view(-1)[idx]


def finalize_descriptors(desc):
    r"""Normalize and reshape descriptors.

    Ref: https://github.com/tkarras/progressive_growing_of_gans/blob/master/metrics/sliced_wasserstein.py  # noqa

    Args:
        desc (list or Tensor): List of descriptors of one level.

    Return:
        Tensor: Descriptors after normalized along channel and flattened.
    """
    if isinstance(desc, list):
        desc = torch.cat(desc, dim=0)
    assert desc.ndim == 4  # (neighborhood, channel, height, width)
    desc -= torch.mean(desc, dim=(0, 2, 3), keepdim=True)
    desc /= torch.std(desc, dim=(0, 2, 3), keepdim=True)
    desc = desc.reshape(desc.shape[0], -1)
    return desc


@METRICS.register_module('SWD')
@METRICS.register_module()
class SlicedWassersteinDistance(GenMetric):
    """SWD (Sliced Wasserstein distance) metric. We calculate the SWD of two
    sets of images in the following way. In every 'feed', we obtain the
    Laplacian pyramids of every images and extract patches from the Laplacian
    pyramids as descriptors. In 'summary', we normalize these descriptors along
    channel, and reshape them so that we can use these descriptors to represent
    the distribution of real/fake images. And we can calculate the sliced
    Wasserstein distance of the real and fake descriptors as the SWD of the
    real and fake images.

    Ref: https://github.com/tkarras/progressive_growing_of_gans/blob/master/metrics/sliced_wasserstein.py # noqa

    Args:
        fake_nums (int): Numbers of the generated image need for the metric.
        image_shape (tuple): Image shape in order "CHW".
        fake_key (Optional[str]): Key for get fake images of the output dict.
            Defaults to None.
        real_key (Optional[str]): Key for get real images from the input dict.
            Defaults to 'gt_img'.
        sample_model (str): Sampling mode for the generative model. Support
            'orig' and 'ema'. Defaults to 'ema'.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    name = 'SWD'

    def __init__(self,
                 fake_nums: int,
                 image_shape: tuple,
                 fake_key: Optional[str] = None,
                 real_key: Optional[str] = 'gt_img',
                 sample_model: str = 'ema',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None):
        super().__init__(fake_nums, fake_nums, fake_key, real_key,
                         sample_model, collect_device, prefix)

        self.nhood_size = 7  # height and width of the extracted patches
        self.nhoods_per_image = 128  # number of extracted patches per image
        self.dir_repeats = 4  # times of sampling directions
        self.dirs_per_repeat = 128  # number of directions per sampling
        self.resolutions = []
        res = image_shape[1]
        self.image_shape = image_shape
        while res >= 16 and len(self.resolutions) < 4:
            self.resolutions.append(res)
            res //= 2
        self.n_pyramids = len(self.resolutions)

        self.gaussian_k = get_gaussian_kernel()
        self.real_results = [[] for res in self.resolutions]
        self.fake_results = [[] for res in self.resolutions]

        self._num_processed = 0

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.fake_results`` and
        ``self.real_results``, which will be used to compute the metrics when
        all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        if self.fake_nums != -1 and (self._num_processed >=
                                     self.fake_nums_per_device):
            return

        real_imgs, fake_imgs = [], []
        for data in data_samples:
            # parse real images
            real_imgs.append(data['gt_img'])
            # parse fake images
            fake_img_ = data
            # get ema/orig results
            if self.sample_model in fake_img_:
                fake_img_ = fake_img_[self.sample_model]
            # get specific fake_keys
            if (self.fake_key is not None and self.fake_key in fake_img_):
                fake_img_ = fake_img_[self.fake_key]
            else:
                # get img tensor
                fake_img_ = fake_img_['fake_img']
            fake_imgs.append(fake_img_)
        real_imgs = torch.stack(real_imgs, dim=0)
        fake_imgs = torch.stack(fake_imgs, dim=0)

        # [0, 255] -> [-1, 1]
        real_imgs = (real_imgs - 127.5) / 127.5
        fake_imgs = (fake_imgs - 127.5) / 127.5

        # real images
        assert real_imgs.shape[1:] == self.image_shape
        if real_imgs.shape[1] == 1:
            real_imgs = real_imgs.repeat(1, 3, 1, 1)
        real_pyramid = laplacian_pyramid(real_imgs, self.n_pyramids - 1,
                                         self.gaussian_k)
        # lod: layer_of_descriptors
        if self.real_results == []:
            self.real_results = [[] for res in self.resolutions]
        for lod, level in enumerate(real_pyramid):
            desc = get_descriptors_for_minibatch(level, self.nhood_size,
                                                 self.nhoods_per_image)
            self.real_results[lod].append(desc.cpu())

        # fake images
        assert fake_imgs.shape[1:] == self.image_shape
        if fake_imgs.shape[1] == 1:
            fake_imgs = fake_imgs.repeat(1, 3, 1, 1)
        fake_pyramid = laplacian_pyramid(fake_imgs, self.n_pyramids - 1,
                                         self.gaussian_k)
        # lod: layer_of_descriptors
        if self.fake_results == []:
            self.fake_results = [[] for res in self.resolutions]
        for lod, level in enumerate(fake_pyramid):
            desc = get_descriptors_for_minibatch(level, self.nhood_size,
                                                 self.nhoods_per_image)
            self.fake_results[lod].append(desc.cpu())

        self._num_processed += real_imgs.shape[0]

    def _collect_target_results(self, target: str) -> Optional[list]:
        """Collect function for SWD metric. This function support collect
        results typing as `List[List[Tensor]]`.

        Args:
            target (str): Target results to collect.

        Returns:
            Optional[list]: The collected results.
        """
        assert target in [
            'fake', 'real'
        ], ('Only support to collect \'fake\' or \'real\' results.')
        results = getattr(self, f'{target}_results')
        results_collected = []
        world_size = get_world_size()
        for result in results:
            # save the original tensor size
            results_size_list = [res.shape[0] for res in result] * world_size
            result_collected = torch.cat(result, dim=0)
            result_collected = torch.cat(all_gather(result_collected), dim=0)
            # split to tuple
            result_collected = torch.split(result_collected, results_size_list)
            # convert to list
            result_collected = [res for res in result_collected]
            results_collected.append(result_collected)

        self._num_processed = 0
        return results_collected

    def compute_metrics(self, results_fake, results_real) -> dict:
        """Compute the result of SWD metric.

        Args:
            fake_results (list): List of image feature of fake images.
            real_results (list): List of image feature of real images.

        Returns:
            dict: A dict of the computed SWD metric.
        """
        fake_descs = [finalize_descriptors(d) for d in results_fake]
        real_descs = [finalize_descriptors(d) for d in results_real]
        distance = [
            sliced_wasserstein(dreal, dfake, self.dir_repeats,
                               self.dirs_per_repeat)
            for dreal, dfake in zip(real_descs, fake_descs)
        ]
        del real_descs
        del fake_descs

        distance = [d * 1e3 for d in distance]  # multiply by 10^3
        result = distance + [np.mean(distance)]

        return {
            f'{resolution}': round(d, 4)
            for resolution, d in zip(self.resolutions + ['avg'], result)
        }
