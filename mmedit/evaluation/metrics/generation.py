# Copyright (c) OpenMMLab. All rights reserved.
import math
import os
import warnings
from abc import ABCMeta
from collections import defaultdict
from copy import deepcopy
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine import print_log
from mmengine.dist import (all_gather, broadcast_object_list, get_world_size,
                           is_main_process)
from PIL import Image
from scipy import linalg
from scipy.stats import entropy
from torch import Tensor
from torch.utils.data.dataloader import DataLoader
from torchvision import models as torchvision_models

from mmedit.models import PerceptualLoss
from mmedit.models.utils import get_module_device
from mmedit.registry import METRICS
from .base_gen_metric import GenMetric
from .inception_utils import (disable_gpu_fuser_on_pt19, load_inception,
                              prepare_inception_feat, prepare_vgg_feat)
from .metric_utils import (compute_pr_distances, finalize_descriptors,
                           get_descriptors_for_minibatch, get_gaussian_kernel,
                           laplacian_pyramid, ms_ssim, slerp,
                           sliced_wasserstein)


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
            Defaults to 'img'.
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
                 real_key: Optional[str] = 'img',
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
        if self._num_processed >= self.fake_nums_per_device:
            return

        real_imgs, fake_imgs = [], []
        for data in data_samples:
            # parse real images
            real_imgs.append(data['gt_img']['data'])
            # parse fake images
            fake_img_ = data
            # get ema/orig results
            if self.sample_model in fake_img_:
                fake_img_ = fake_img_[self.sample_model]
            # get specific fake_keys
            if (self.fake_key is not None and self.fake_key in fake_img_):
                fake_img_ = fake_img_[self.fake_key]['data']
            else:
                # get img tensor
                fake_img_ = fake_img_['fake_img']['data']
            fake_imgs.append(fake_img_)
        real_imgs = torch.stack(real_imgs, dim=0)
        fake_imgs = torch.stack(fake_imgs, dim=0)

        # real images
        assert real_imgs.shape[1:] == self.image_shape
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
        """Compulate the result of SWD metric.

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


class GenerativeMetric(GenMetric, metaclass=ABCMeta):
    """Metric for generative metrics. Except for the preparation phase
    (:meth:`prepare`), generative metrics do not need extra real images.

    Args:
        fake_nums (int): Numbers of the generated image need for the metric.
        real_nums (int): Numbers of the real image need for the metric. If `-1`
            is passed means all images from the dataset is need. Defaults to 0.
        fake_key (Optional[str]): Key for get fake images of the output dict.
            Defaults to None.
        real_key (Optional[str]): Key for get real images from the input dict.
            Defaults to 'img'.
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
    SAMPLER_MODE = 'Generative'

    def __init__(self,
                 fake_nums: int,
                 real_nums: int = 0,
                 fake_key: Optional[str] = None,
                 real_key: Optional[str] = 'img',
                 sample_model: str = 'ema',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None):
        super().__init__(fake_nums, real_nums, fake_key, real_key,
                         sample_model, collect_device, prefix)

    def get_metric_sampler(self, model: nn.Module, dataloader: DataLoader,
                           metrics: GenMetric):
        """Get sampler for generative metrics. Returns a dummy iterator, whose
        return value of each iteration is a dict containing batch size and
        sample mode to generate images.

        Args:
            model (nn.Module): Model to evaluate.
            dataloader (DataLoader): Dataloader for real images. Used to get
                batch size during generate fake images.
            metrics (List['GenMetric']): Metrics with the same sampler mode.

        Returns:
            :class:`dummy_iterator`: Sampler for generative metrics.
        """

        batch_size = dataloader.batch_size

        sample_model = metrics[0].sample_model
        assert all([metric.sample_model == sample_model for metric in metrics
                    ]), ('\'sample_model\' between metrics is inconsistency.')

        class dummy_iterator:

            def __init__(self, batch_size, max_length, sample_model) -> None:
                self.batch_size = batch_size
                self.max_length = max_length
                self.sample_model = sample_model

            def __iter__(self) -> Iterator:
                self.idx = 0
                return self

            def __len__(self) -> int:
                return math.ceil(self.max_length / self.batch_size)

            def __next__(self) -> dict:
                if self.idx > self.max_length:
                    raise StopIteration
                self.idx += batch_size
                return dict(
                    inputs=dict(
                        sample_model=self.sample_model,
                        num_batches=self.batch_size))

        return dummy_iterator(
            batch_size=batch_size,
            max_length=max([metric.fake_nums_per_device
                            for metric in metrics]),
            sample_model=sample_model)

    def evaluate(self) -> dict():
        """Evaluate generative metric. In this function we only collect
        :attr:`fake_results` because generative metrics do not need real
        images.

        Returns:
            dict: Evaluation metrics dict on the val dataset. The keys are the
                names of the metrics, and the values are corresponding results.
        """
        results_fake = self._collect_target_results(target='fake')

        if is_main_process():
            # pack to list, align with BaseMetrics
            _metrics = self.compute_metrics(results_fake)
            # Add prefix to metric names
            if self.prefix:
                _metrics = {
                    '/'.join((self.prefix, k)): v
                    for k, v in _metrics.items()
                }
            metrics = [_metrics]
        else:
            metrics = [None]  # type: ignore

        broadcast_object_list(metrics)

        # reset the results list
        self.fake_results.clear()

        return metrics[0]

    def compute_metrics(self, results) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """


@METRICS.register_module('FID-Full')
@METRICS.register_module('FID')
@METRICS.register_module()
class FrechetInceptionDistance(GenerativeMetric):
    """FID metric. In this metric, we calculate the distance between real
    distributions and fake distributions. The distributions are modeled by the
    real samples and fake samples, respectively. `Inception_v3` is adopted as
    the feature extractor, which is widely used in StyleGAN and BigGAN.

    Args:
        fake_nums (int): Numbers of the generated image need for the metric.
        real_nums (int): Numbers of the real images need for the metric. If -1
            is passed, means all real images in the dataset will be used.
            Defaults to -1.
        inception_style (str): The target inception style want to load. If the
            given style cannot be loaded successful, will attempt to load a
            valid one. Defaults to 'StyleGAN'.
        inception_path (str, optional): Path the the pretrain Inception
            network. Defaults to None.
        inception_pkl (str, optional): Path to reference inception pickle file.
            If `None`, the statistical value of real distribution will be
            calculated at running time. Defaults to None.
        fake_key (Optional[str]): Key for get fake images of the output dict.
            Defaults to None.
        real_key (Optional[str]): Key for get real images from the input dict.
            Defaults to 'img'.
        sample_model (str): Sampling mode for the generative model. Support
            'orig' and 'ema'. Defaults to 'orig'.
        collect_device (str, optional): Device name used for collecting results
            from different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """
    name = 'FID'

    def __init__(self,
                 fake_nums: int,
                 real_nums: int = -1,
                 inception_style='StyleGAN',
                 inception_path: Optional[str] = None,
                 inception_pkl: Optional[str] = None,
                 fake_key: Optional[str] = None,
                 real_key: Optional[str] = 'img',
                 sample_model: str = 'orig',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None):
        super().__init__(fake_nums, real_nums, fake_key, real_key,
                         sample_model, collect_device, prefix)
        self.real_mean = None
        self.real_cov = None
        self.device = 'cpu'
        self.inception, self.inception_style = self._load_inception(
            inception_style, inception_path)
        self.inception_pkl = inception_pkl

    def prepare(self, module: nn.Module, dataloader: DataLoader) -> None:
        """Preparing inception feature for the real images.

        Args:
            module (nn.Module): The model to evaluate.
            dataloader (DataLoader): The dataloader for real images.
        """
        self.device = module.data_preprocessor.device
        self.inception.to(self.device)
        inception_feat_dict = prepare_inception_feat(
            dataloader, self, module.data_preprocessor, capture_mean_cov=True)
        if is_main_process():
            self.real_mean = inception_feat_dict['real_mean']
            self.real_cov = inception_feat_dict['real_cov']

    def _load_inception(self, inception_style: str,
                        inception_path: Optional[str]
                        ) -> Tuple[nn.Module, str]:
        """Load inception and return the successful loaded style.

        Args:
            inception_style (str): Target style of Inception network want to
                load.
            inception_path (Optional[str]): The path to the inception.

        Returns:
            Tuple[nn.Module, str]: The actually loaded inception network and
                corresponding style.
        """
        if inception_style == 'StyleGAN':
            args = dict(type='StyleGAN', inception_path=inception_path)
        else:
            args = dict(type='Pytorch', normalize_input=False)
        inception, style = load_inception(args, 'FID')
        inception.eval()
        return inception, style

    def forward_inception(self, image: Tensor) -> Tensor:
        """Feed image to inception network and get the output feature.

        Args:
            image (Tensor): Image tensor fed to the Inception network.

        Returns:
            Tensor: Image feature extracted from inception.
        """

        # image must passed with 'bgr'
        image = image[:, [2, 1, 0]]
        image = image.to(self.device)
        if self.inception_style == 'StyleGAN':
            image = (image * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            with disable_gpu_fuser_on_pt19():
                feat = self.inception(image, return_features=True)
        else:
            feat = self.inception(image)[0].view(image.shape[0], -1)
        return feat

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.fake_results``, which will be used
        to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        if len(self.fake_results) >= self.fake_nums_per_device:
            return

        fake_imgs = []
        for pred in data_samples:
            fake_img_ = pred
            # get ema/orig results
            if self.sample_model in fake_img_:
                fake_img_ = fake_img_[self.sample_model]
            # get specific fake_keys
            if (self.fake_key is not None and self.fake_key in fake_img_):
                fake_img_ = fake_img_[self.fake_key]['data']
            else:
                # get img tensor
                fake_img_ = fake_img_['fake_img']['data']
            fake_imgs.append(fake_img_)
        fake_imgs = torch.stack(fake_imgs, dim=0)

        feat = self.forward_inception(fake_imgs)
        feat_list = list(torch.split(feat, 1))
        self.fake_results += feat_list

    @staticmethod
    def _calc_fid(sample_mean: np.ndarray,
                  sample_cov: np.ndarray,
                  real_mean: np.ndarray,
                  real_cov: np.ndarray,
                  eps: float = 1e-6) -> Tuple[float]:
        """Refer to the implementation from:

        https://github.com/rosinality/stylegan2-pytorch/blob/master/fid.py#L34
        """
        cov_sqrt, _ = linalg.sqrtm(sample_cov @ real_cov, disp=False)

        if not np.isfinite(cov_sqrt).all():
            print('product of cov matrices is singular')
            offset = np.eye(sample_cov.shape[0]) * eps
            cov_sqrt = linalg.sqrtm(
                (sample_cov + offset) @ (real_cov + offset))

        if np.iscomplexobj(cov_sqrt):
            if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
                m = np.max(np.abs(cov_sqrt.imag))

                raise ValueError(f'Imaginary component {m}')

            cov_sqrt = cov_sqrt.real

        mean_diff = sample_mean - real_mean
        mean_norm = mean_diff @ mean_diff

        trace = np.trace(sample_cov) + np.trace(
            real_cov) - 2 * np.trace(cov_sqrt)

        fid = mean_norm + trace

        return float(fid), float(mean_norm), float(trace)

    def compute_metrics(self, fake_results: list) -> dict:
        """Compulate the result of FID metric.

        Args:
            fake_results (list): List of image feature of fake images.

        Returns:
            dict: A dict of the computed FID metric and its mean and
                covariance.
        """
        fake_feats = torch.cat(fake_results, dim=0)
        fake_feats_np = fake_feats.cpu().numpy()
        fake_mean = np.mean(fake_feats_np, 0)
        fake_cov = np.cov(fake_feats_np, rowvar=False)

        fid, mean, cov = self._calc_fid(fake_mean, fake_cov, self.real_mean,
                                        self.real_cov)

        return {'fid': fid, 'mean': mean, 'cov': cov}


@METRICS.register_module('IS')
@METRICS.register_module()
class InceptionScore(GenerativeMetric):
    """IS (Inception Score) metric. The images are split into groups, and the
    inception score is calculated on each group of images, then the mean and
    standard deviation of the score is reported. The calculation of the
    inception score on a group of images involves first using the inception v3
    model to calculate the conditional probability for each image (p(y|x)). The
    marginal probability is then calculated as the average of the conditional
    probabilities for the images in the group (p(y)). The KL divergence is then
    calculated for each image as the conditional probability multiplied by the
    log of the conditional probability minus the log of the marginal
    probability. The KL divergence is then summed over all images and averaged
    over all classes and the exponent of the result is calculated to give the
    final score.

    Ref: https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py  # noqa

    Note that we highly recommend that users should download the Inception V3
    script module from the following address. Then, the `inception_pkl` can
    be set with user's local path. If not given, we will use the Inception V3
    from pytorch model zoo. However, this may bring significant different in
    the final results.

    Tero's Inception V3: https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt  # noqa

    Args:
        fake_nums (int): Numbers of the generated image need for the metric.
        resize (bool, optional): Whether resize image to 299x299. Defaults to
            True.
        splits (int, optional): The number of groups. Defaults to 10.
        inception_style (str): The target inception style want to load. If the
            given style cannot be loaded successful, will attempt to load a
            valid one. Defaults to 'StyleGAN'.
        inception_path (str, optional): Path the the pretrain Inception
            network. Defaults to None.
        resize_method (str): Resize method. If `resize` is False, this will be
            ignored. Defaults to 'bicubic'.
        use_pil_resize (bool): Whether use Bicubic interpolation with
            Pillow's backend. If set as True, the evaluation process may be a
            little bit slow, but achieve a more accurate IS result. Defaults
            to False.
        fake_key (Optional[str]): Key for get fake images of the output dict.
            Defaults to None.
        real_key (Optional[str]): Key for get real images from the input dict.
            Defaults to 'img'.
        sample_model (str): Sampling mode for the generative model. Support
            'orig' and 'ema'. Defaults to 'orig'.
        collect_device (str, optional): Device name used for collecting results
            from different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """
    name = 'IS'

    pil_resize_method_mapping = {
        'bicubic': Image.BICUBIC,
        'bilinear': Image.BILINEAR,
        'nearest': Image.NEAREST,
        'box': Image.BOX
    }

    def __init__(self,
                 fake_nums: int = 5e4,
                 resize: bool = True,
                 splits: int = 10,
                 inception_style: str = 'StyleGAN',
                 inception_path: Optional[str] = None,
                 resize_method='bicubic',
                 use_pillow_resize: bool = True,
                 fake_key: Optional[str] = None,
                 sample_model='orig',
                 collect_device: str = 'cpu',
                 prefix: str = None):
        super().__init__(fake_nums, 0, fake_key, None, sample_model,
                         collect_device, prefix)

        self.resize = resize
        self.splits = splits
        self.device = 'cpu'

        if not use_pillow_resize:
            print_log(
                'We strongly recommend to use the bicubic resize with '
                'Pillow backend. Otherwise, the results maybe '
                'unreliable', 'current')
        self.use_pillow_resize = use_pillow_resize

        if self.use_pillow_resize:
            allowed_resize_method = list(self.pil_resize_method_mapping.keys())
            assert resize_method in self.pil_resize_method_mapping, (
                f'\'resize_method\' (\'{resize_method}\') is not supported '
                'for PIL resize. Please select resize method in '
                f'{allowed_resize_method}.')
            self.resize_method = self.pil_resize_method_mapping[resize_method]
        else:
            self.resize_method = resize_method

        self.inception, self.inception_style = self._load_inception(
            inception_style, inception_path)

    def prepare(self, module: nn.Module, dataloader: DataLoader) -> None:
        """Prepare for the pre-calculating items of the metric. Defaults to do
        nothing.

        Args:
            module (nn.Module): Model to evaluate.
            dataloader (DataLoader): Dataloader for the real images.
        """
        self.device = module.data_preprocessor.device
        self.inception.to(self.device)

    def _load_inception(self, inception_style: str,
                        inception_path: Optional[str]
                        ) -> Tuple[nn.Module, str]:
        """Load pretrain model of inception network.
        Args:
            inception_style (str): Target style of Inception network want to
                load.
            inception_path (Optional[str]): The path to the inception.

        Returns:
            Tuple[nn.Module, str]: The actually loaded inception network and
                corresponding style.
        """
        inception, style = load_inception(
            dict(type=inception_style, inception_path=inception_path), 'IS')
        inception.eval()
        return inception, style

    def _preprocess(self, image: Tensor) -> Tensor:
        """Preprocess image before pass to the Inception. Preprocess operations
        contain channel conversion and resize.

        Args:
            image (Tensor): Image tensor before preprocess.

        Returns:
            Tensor: Image tensor after resize and channel conversion
                (if need.)
        """
        # image must passed in 'bgr'
        image = image[:, [2, 1, 0]]
        if not self.resize:
            return image
        if self.use_pillow_resize:
            image = (image.clone() * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            x_np = [x_.permute(1, 2, 0).detach().cpu().numpy() for x_ in image]

            # use bicubic resize as default
            x_pil = [
                Image.fromarray(x_).resize((299, 299),
                                           resample=self.resize_method)
                for x_ in x_np
            ]
            x_ten = torch.cat(
                [torch.FloatTensor(np.array(x_)[None, ...]) for x_ in x_pil])
            x_ten = (x_ten / 127.5 - 1).to(torch.float)
            return x_ten.permute(0, 3, 1, 2)
        else:
            return F.interpolate(
                image, size=(299, 299), mode=self.resize_method)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.fake_results``, which will be used
        to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        if len(self.fake_results) >= self.fake_nums_per_device:
            return

        fake_imgs = []
        for pred in data_samples:
            fake_img_ = pred
            # get ema/orig results
            if self.sample_model in fake_img_:
                fake_img_ = fake_img_[self.sample_model]
            # get specific fake_keys
            if (self.fake_key is not None and self.fake_key in fake_img_):
                fake_img_ = fake_img_[self.fake_key]['data']
            else:
                # get img tensor
                fake_img_ = fake_img_['fake_img']['data']
            fake_imgs.append(fake_img_)
        fake_imgs = torch.stack(fake_imgs, dim=0)
        fake_imgs = self._preprocess(fake_imgs).to(self.device)

        if self.inception_style == 'StyleGAN':
            fake_imgs = (fake_imgs * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            with disable_gpu_fuser_on_pt19():
                feat = self.inception(fake_imgs, no_output_bias=True)
        else:
            feat = F.softmax(self.inception(fake_imgs), dim=1)

        # NOTE: feat is shape like (bz, 1000), convert to a list
        self.fake_results += list(torch.split(feat, 1))

    def compute_metrics(self, fake_results: list) -> dict:
        """Compute the results of Inception Score metric.

        Args:
            fake_results (list): List of image feature of fake images.

        Returns:
            dict: A dict of the computed IS metric and its standard error
        """
        split_scores = []
        preds = torch.cat(fake_results, dim=0).cpu().numpy()
        # check for the size
        assert preds.shape[0] >= self.fake_nums
        preds = preds[:self.fake_nums]
        for k in range(self.splits):
            part = preds[k * (self.fake_nums // self.splits):(k + 1) *
                         (self.fake_nums // self.splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))

        mean, std = np.mean(split_scores), np.std(split_scores)

        return {'is': float(mean), 'is_std': float(std)}


@METRICS.register_module('MS_SSIM')
@METRICS.register_module()
class MultiScaleStructureSimilarity(GenerativeMetric):
    """MS-SSIM (Multi-Scale Structure Similarity) metric.

    Ref: https://github.com/tkarras/progressive_growing_of_gans/blob/master/metrics/ms_ssim.py # noqa

    Args:
        fake_nums (int): Numbers of the generated image need for the metric.
        fake_key (Optional[str]): Key for get fake images of the output dict.
            Defaults to None.
        real_key (Optional[str]): Key for get real images from the input dict.
            Defaults to 'img'.
        sample_model (str): Sampling mode for the generative model. Support
            'orig' and 'ema'. Defaults to 'ema'.
        collect_device (str, optional): Device name used for collecting results
            from different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """
    name = 'MS-SSIM'

    def __init__(self,
                 fake_nums: int,
                 fake_key: Optional[str] = None,
                 sample_model: str = 'ema',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(fake_nums, 0, fake_key, None, sample_model,
                         collect_device, prefix)

        assert fake_nums % 2 == 0
        self.num_pairs = fake_nums // 2

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Feed data to the metric.

        Args:
            data_batch (dict): Real images from dataloader. Do not be used
                in this metric.
            data_samples (Sequence[dict]): Generated images.
        """
        if len(self.fake_results) >= (self.fake_nums_per_device // 2):
            return

        fake_imgs = []
        for pred in data_samples:
            fake_img_ = pred
            # get ema/orig results
            if self.sample_model in fake_img_:
                fake_img_ = fake_img_[self.sample_model]
            # get specific fake_keys
            if (self.fake_key is not None and self.fake_key in fake_img_):
                fake_img_ = fake_img_[self.fake_key]['data']
            else:
                # get img tensor
                fake_img_ = fake_img_['fake_img']['data']
            fake_imgs.append(fake_img_)
        minibatch = torch.stack(fake_imgs, dim=0)

        assert minibatch.shape[0] % 2 == 0, 'batch size must be divided by 2.'
        minibatch = ((minibatch + 1) / 2)
        minibatch = minibatch.clamp_(0, 1)
        half1 = minibatch[0::2].cpu().data.numpy().transpose((0, 2, 3, 1))
        half1 = (half1 * 255).astype('uint8')
        half2 = minibatch[1::2].cpu().data.numpy().transpose((0, 2, 3, 1))
        half2 = (half2 * 255).astype('uint8')

        scores = ms_ssim(half1, half2, reduce_mean=False)
        self.fake_results += [torch.Tensor([s]) for s in scores.tolist()]

    def _collect_target_results(self, target: str) -> Optional[list]:
        """Collected results for MS-SSIM metric. Size of `self.fake_results` in
        MS-SSIM does not relay on `self.fake_nums` but `self.num_pairs`.

        Args:
            target (str): Target results to collect.

        Returns:
            Optional[list]: The collected results.
        """
        assert target in 'fake', 'Only support to collect \'fake\' results.'
        results = getattr(self, f'{target}_results')
        size = self.num_pairs
        size = len(results) * get_world_size() if size == -1 else size

        if len(results) == 0:
            warnings.warn(
                f'{self.__class__.__name__} got empty `self.{target}_results`.'
                ' Please ensure that the processed results are properly added '
                f'into `self.{target}_results` in `process` method.')

        # apply all_gather for tensor results
        results = torch.cat(results, dim=0)
        results = torch.cat(all_gather(results), dim=0)[:size]
        results = torch.split(results, 1)

        # on non-main process, results should be `None`
        if is_main_process() and len(results) != size:
            raise ValueError(f'Length of results is \'{len(results)}\', not '
                             f'equals to target size \'{size}\'.')
        return results

    def compute_metrics(self, results_fake: List):
        """Computed the result of MS-SSIM.

        Returns:
            dict: Calculated MS-SSIM result.
        """
        results = torch.cat(results_fake, dim=0)
        avg = results.sum() / self.num_pairs
        return {'avg': round(avg.item(), 4)}


@METRICS.register_module('PR')
@METRICS.register_module()
class PrecisionAndRecall(GenerativeMetric):
    r"""Improved Precision and recall metric.

        In this metric, we draw real and generated samples respectively, and
        embed them into a high-dimensional feature space using a pre-trained
        classifier network. We use these features to estimate the corresponding
        manifold. We obtain the estimation by calculating pairwise Euclidean
        distances between all feature vectors in the set and, for each feature
        vector, construct a hypersphere with radius equal to the distance to its
        kth nearest neighbor. Together, these hyperspheres define a volume in
        the feature space that serves as an estimate of the true manifold.
        Precision is quantified by querying for each generated image whether
        the image is within the estimated manifold of real images.
        Symmetrically, recall is calculated by querying for each real image
        whether the image is within estimated manifold of generated image.

        Ref: https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/metrics/precision_recall.py  # noqa

        Note that we highly recommend that users should download the vgg16
        script module from the following address. Then, the `vgg16_script` can
        be set with user's local path. If not given, we will use the vgg16 from
        pytorch model zoo. However, this may bring significant different in the
        final results.

        Tero's vgg16: https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt

        Args:
            num_images (int): The number of evaluated generated samples.
            image_shape (tuple): Image shape in order "CHW". Defaults to None.
            num_real_need (int | None, optional): The number of real images.
                Defaults to None.
            full_dataset (bool, optional): Whether to use full dataset for
                evaluation. Defaults to False.
            k (int, optional): Kth nearest parameter. Defaults to 3.
            bgr2rgb (bool, optional): Whether to change the order of image
                channel. Defaults to True.
            vgg16_script (str, optional): Path for the Tero's vgg16 module.
                Defaults to 'work_dirs/cache/vgg16.pt'.
            row_batch_size (int, optional): The batch size of row data.
                Defaults to 10000.
            col_batch_size (int, optional): The batch size of col data.
                Defaults to 10000.
            auto_save (bool, optional): Whether save vgg feature automatically.
        """
    name = 'PR'

    def __init__(self,
                 fake_nums,
                 real_nums=-1,
                 k=3,
                 fake_key: Optional[str] = None,
                 real_key: Optional[str] = 'img',
                 sample_model: str = 'ema',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 vgg16_script='work_dirs/cache/vgg16.pt',
                 vgg16_pkl=None,
                 row_batch_size=10000,
                 col_batch_size=10000,
                 auto_save=True):
        super().__init__(fake_nums, real_nums, fake_key, real_key,
                         sample_model, collect_device, prefix)
        print_log('loading vgg16 for improved precision and recall...',
                  'current')
        self.vgg16_pkl = vgg16_pkl
        self.vgg16, self.use_tero_scirpt = self._load_vgg(vgg16_script)
        self.k = k

        self.auto_save = auto_save
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size

    def _load_vgg(self, vgg16_script: Optional[str]) -> Tuple[nn.Module, bool]:
        """Load VGG network from the given path.

        Args:
            vgg16_script: The path of script model of VGG network. If None,
                will load the pytorch version.

        Returns:
            Tuple[nn.Module, str]: The actually loaded VGG network and
                corresponding style.
        """
        if os.path.isfile(vgg16_script):
            vgg16 = torch.jit.load('work_dirs/cache/vgg16.pt').eval()
            use_tero_scirpt = True
        else:
            print_log(
                'Cannot load Tero\'s script module. Use official '
                'vgg16 instead', 'current')
            vgg16 = torchvision_models.vgg16(pretrained=True).eval()
            use_tero_scirpt = False
        return vgg16, use_tero_scirpt

    @torch.no_grad()
    def extract_features(self, images):
        """Extracting image features.

        Args:
            images (torch.Tensor): Images tensor.
        Returns:
            torch.Tensor: Vgg16 features of input images.
        """
        # image must passed in 'bgr'
        images = images[:, [2, 1, 0], ...]
        if self.use_tero_scirpt:
            images = (images * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            feature = self.vgg16(images, return_features=True)
        else:
            batch = F.interpolate(images, size=(224, 224))
            before_fc = self.vgg16.features(batch)
            before_fc = before_fc.view(-1, 7 * 7 * 512)
            feature = self.vgg16.classifier[:4](before_fc)

        return feature

    @torch.no_grad()
    def compute_metrics(self, results_fake) -> dict:
        """compute_metrics.

        Returns:
            dict: Summarized results.
        """
        gen_features = torch.cat(results_fake, dim=0).to(self.collect_device)
        real_features = self.results_real

        self._result_dict = {}

        for name, manifold, probes in [
            ('precision', real_features, gen_features),
            ('recall', gen_features, real_features)
        ]:
            kth = []
            for manifold_batch in manifold.split(self.row_batch_size):
                distance = compute_pr_distances(
                    row_features=manifold_batch,
                    col_features=manifold,
                    col_batch_size=self.col_batch_size)
                kth.append(
                    distance.to(torch.float32).kthvalue(self.k + 1).values.to(
                        torch.float16))
            kth = torch.cat(kth)
            pred = []
            for probes_batch in probes.split(self.row_batch_size):
                distance = compute_pr_distances(
                    row_features=probes_batch,
                    col_features=manifold,
                    col_batch_size=self.col_batch_size)
                pred.append((distance <= kth).any(dim=1))
            self._result_dict[name] = float(
                torch.cat(pred).to(torch.float32).mean())

        precision = self._result_dict['precision']
        recall = self._result_dict['recall']
        self._result_str = f'precision: {precision}, recall:{recall}'
        return self._result_dict

    @torch.no_grad()
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.fake_results``, which will be used
        to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        fake_imgs = []
        for pred in data_samples:
            fake_img_ = pred
            # get ema/orig results
            if self.sample_model in fake_img_:
                fake_img_ = fake_img_[self.sample_model]
            # get specific fake_keys
            if (self.fake_key is not None and self.fake_key in fake_img_):
                fake_img_ = fake_img_[self.fake_key]['data']
            else:
                # get img tensor
                fake_img_ = fake_img_['fake_img']['data']
            fake_imgs.append(fake_img_)
        fake_imgs = torch.stack(fake_imgs, dim=0)
        feat = self.extract_features(fake_imgs)
        feat_list = list(torch.split(feat, 1))
        self.fake_results += feat_list

    @torch.no_grad()
    def prepare(self, module: nn.Module, dataloader: DataLoader) -> None:
        # move to corresponding device
        device = get_module_device(module)
        self.vgg16.to(device)

        vgg_feat = prepare_vgg_feat(dataloader, self, module.data_preprocessor,
                                    self.auto_save)
        if self.real_nums != -1:
            assert self.real_nums <= vgg_feat.shape[0], (
                f'Need \'{self.real_nums}\' of real nums, but only '
                f'\'{vgg_feat.shape[0]}\' images be found in the '
                'inception feature.')
            vgg_feat = vgg_feat[np.random.choice(
                vgg_feat.shape[0], size=self.real_nums, replace=True)]
        self.results_real = vgg_feat


@METRICS.register_module('EQ')
@METRICS.register_module()
class Equivariance(GenerativeMetric):

    name = 'Equivariance'

    def __init__(self,
                 fake_nums: int,
                 real_nums: int = 0,
                 fake_key: Optional[str] = None,
                 real_key: Optional[str] = 'img',
                 sample_mode: str = 'ema',
                 sample_kwargs: dict = dict(),
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 eq_cfg=dict()):
        super().__init__(fake_nums, real_nums, fake_key, real_key, sample_mode,
                         collect_device, prefix)
        # set default sampler config
        self._eq_cfg = deepcopy(eq_cfg)
        self._eq_cfg.setdefault('compute_eqt_int', False)
        self._eq_cfg.setdefault('compute_eqt_frac', False)
        self._eq_cfg.setdefault('compute_eqr', False)
        self._eq_cfg.setdefault('translate_max', 0.125)
        self._eq_cfg.setdefault('rotate_max', 1)
        self.SAMPLER_MODE = 'EqSampler'

        self.sample_kwargs = sample_kwargs
        # compute numbers of eq
        self.n_sub_metric = 0
        if self._eq_cfg['compute_eqt_int']:
            self.n_sub_metric += 1
        if self._eq_cfg['compute_eqt_frac']:
            self.n_sub_metric += 1
        if self._eq_cfg['compute_eqr']:
            self.n_sub_metric += 1

        self.fake_results = defaultdict(list)

    @torch.no_grad()
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.fake_results``, which will be used
        to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        cfg_key_list = ['compute_eqt_int', 'compute_eqt_frac', 'compute_eqr']
        sample_key_list = ['eqt_int', 'eqt_frac', 'eqr']
        for pred in data_samples:
            for cfg_key, sample_key in zip(cfg_key_list, sample_key_list):
                if self._eq_cfg[cfg_key]:
                    assert sample_key in pred
                    # assert hasattr(pred, sample_key)
                    eq_sample = pred[sample_key]
                    diff = eq_sample['diff'].to(torch.float64).sum()
                    mask = eq_sample['mask'].to(torch.float64).sum()
                    self.fake_results[sample_key] += [diff, mask]

    def get_metric_sampler(self, model: nn.Module, dataloader: DataLoader,
                           metrics: GenMetric):
        """Get sampler for generative metrics. Returns a dummy iterator, whose
        return value of each iteration is a dict containing batch size and
        sample mode to generate images.

        Args:
            model (nn.Module): Model to evaluate.
            dataloader (DataLoader): Dataloader for real images. Used to get
                batch size during generate fake images.
            metrics (List['GenMetric']): Metrics with the same sampler mode.

        Returns:
            :class:`dummy_iterator`: Sampler for generative metrics.
        """

        batch_size = dataloader.batch_size

        sample_model = metrics[0].sample_model
        assert all([metric.sample_model == sample_model for metric in metrics
                    ]), ('\'sample_model\' between metrics is inconsistency.')

        return eq_iterator(
            batch_size=batch_size,
            max_length=max([metric.fake_nums_per_device
                            for metric in metrics]),
            sample_mode=sample_model,
            eq_cfg=self._eq_cfg,
            sample_kwargs=self.sample_kwargs)

    def compute_metrics(self, results) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        results = dict()
        for key in ['eqt_int', 'eqt_frac', 'eqr']:
            if key not in self.fake_results:
                continue
            sums = torch.stack(self.fake_results[key], dim=0)
            mses = (sums[0::2] / sums[1::2]).mean()
            psnrs = np.log10(2) * 20 - mses.log10() * 10
            psnrs = psnrs.cpu().numpy()
            results[key] = psnrs
        return results

    def _collect_target_results(self, target: str) -> Optional[list]:
        """Collect function for Eq metric. This function support collect
        results typing as Dict[List[Tensor]]`.

        Args:
            target (str): Target results to collect.

        Returns:
            Optional[list]: The collected results.
        """
        if target == 'real':
            return
        results = getattr(self, f'{target}_results')
        results_collected = []
        results_collected = dict()
        for key, result in results.items():
            result_collected = torch.stack(result)
            result_collected = torch.cat(all_gather(result_collected), dim=0)
            results_collected[key] = torch.split(result_collected,
                                                 len(result_collected))

        return results_collected


class eq_iterator:

    def __init__(self, batch_size, max_length, sample_mode, eq_cfg,
                 sample_kwargs) -> None:
        self.batch_size = batch_size
        self.max_length = max_length
        self.sample_mode = sample_mode
        self.eq_cfg = deepcopy(eq_cfg)
        self.sample_kwargs = sample_kwargs

    def __iter__(self) -> Iterator:
        self.idx = 0
        return self

    def __len__(self) -> int:
        return self.max_length // self.batch_size

    def __next__(self) -> dict:
        if self.idx >= self.max_length:
            raise StopIteration
        self.idx += self.batch_size
        mode = dict(
            sample_mode=self.sample_mode,
            eq_cfg=self.eq_cfg,
            sample_kwargs=self.sample_kwargs)
        # StyleGAN3 forward will receive eq config from mode
        return dict(inputs=dict(mode=mode, num_batches=self.batch_size))


@METRICS.register_module()
class TransFID(FrechetInceptionDistance):

    def __init__(self,
                 fake_nums: int,
                 real_nums: int = -1,
                 inception_style='StyleGAN',
                 inception_path: Optional[str] = None,
                 inception_pkl: Optional[str] = None,
                 fake_key: Optional[str] = None,
                 real_key: Optional[str] = 'img',
                 sample_model: str = 'ema',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None):
        super().__init__(fake_nums, real_nums, inception_style, inception_path,
                         inception_pkl, fake_key, real_key, sample_model,
                         collect_device, prefix)

        self.SAMPLER_MODE = 'normal'

    def get_metric_sampler(self, model: nn.Module, dataloader: DataLoader,
                           metrics: List['GenMetric']) -> DataLoader:
        """Get sampler for normal metrics. Directly returns the dataloader.

        Args:
            model (nn.Module): Model to evaluate.
            dataloader (DataLoader): Dataloader for real images.
            metrics (List['GenMetric']): Metrics with the same sample mode.

        Returns:
            DataLoader: Default sampler for normal metrics.
        """
        return dataloader

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.fake_results``, which will be used
        to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        fake_imgs = []
        for pred in data_samples:
            fake_img_ = pred
            # get ema/orig results
            if self.sample_model in fake_img_:
                fake_img_ = fake_img_[self.sample_model]
            # get specific fake_keys
            if (self.fake_key is not None and self.fake_key in fake_img_):
                fake_img_ = fake_img_[self.fake_key]
            # get img tensor
            fake_img_ = fake_img_['data']
            fake_imgs.append(fake_img_)
        fake_imgs = torch.stack(fake_imgs, dim=0)
        feat = self.forward_inception(fake_imgs)
        feat_list = list(torch.split(feat, 1))
        self.fake_results += feat_list


@METRICS.register_module()
class TransIS(InceptionScore):
    """IS (Inception Score) metric. The images are split into groups, and the
    inception score is calculated on each group of images, then the mean and
    standard deviation of the score is reported. The calculation of the
    inception score on a group of images involves first using the inception v3
    model to calculate the conditional probability for each image (p(y|x)). The
    marginal probability is then calculated as the average of the conditional
    probabilities for the images in the group (p(y)). The KL divergence is then
    calculated for each image as the conditional probability multiplied by the
    log of the conditional probability minus the log of the marginal
    probability. The KL divergence is then summed over all images and averaged
    over all classes and the exponent of the result is calculated to give the
    final score.

    Ref: https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py  # noqa

    Note that we highly recommend that users should download the Inception V3
    script module from the following address. Then, the `inception_pkl` can
    be set with user's local path. If not given, we will use the Inception V3
    from pytorch model zoo. However, this may bring significant different in
    the final results.

    Tero's Inception V3: https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt  # noqa

    Args:
        fake_nums (int): Numbers of the generated image need for the metric.
        resize (bool, optional): Whether resize image to 299x299. Defaults to
            True.
        splits (int, optional): The number of groups. Defaults to 10.
        inception_style (str): The target inception style want to load. If the
            given style cannot be loaded successful, will attempt to load a
            valid one. Defaults to 'StyleGAN'.
        inception_path (str, optional): Path the the pretrain Inception
            network. Defaults to None.
        resize_method (str): Resize method. If `resize` is False, this will be
            ignored. Defaults to 'bicubic'.
        use_pil_resize (bool): Whether use Bicubic interpolation with
            Pillow's backend. If set as True, the evaluation process may be a
            little bit slow, but achieve a more accurate IS result. Defaults
            to False.
        fake_key (Optional[str]): Key for get fake images of the output dict.
            Defaults to None.
        real_key (Optional[str]): Key for get real images from the input dict.
            Defaults to 'img'.
        sample_model (str): Sampling mode for the generative model. Support
            'orig' and 'ema'. Defaults to 'ema'.
        collect_device (str, optional): Device name used for collecting results
            from different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    def __init__(self,
                 fake_nums: int = 50000,
                 resize: bool = True,
                 splits: int = 10,
                 inception_style: str = 'StyleGAN',
                 inception_path: Optional[str] = None,
                 resize_method='bicubic',
                 use_pillow_resize: bool = True,
                 fake_key: Optional[str] = None,
                 sample_model='ema',
                 collect_device: str = 'cpu',
                 prefix: str = None):
        super().__init__(fake_nums, resize, splits, inception_style,
                         inception_path, resize_method, use_pillow_resize,
                         fake_key, sample_model, collect_device, prefix)
        self.SAMPLER_MODE = 'normal'

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.fake_results``, which will be used
        to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            predictions (Sequence[dict]): A batch of outputs from the model.
        """
        if len(self.fake_results) >= self.fake_nums_per_device:
            return

        fake_imgs = []
        for pred in data_samples:
            fake_img_ = pred
            # get ema/orig results
            if self.sample_model in fake_img_:
                fake_img_ = fake_img_[self.sample_model]
            # get specific fake_keys
            if (self.fake_key is not None and self.fake_key in fake_img_):
                fake_img_ = fake_img_[self.fake_key]
            # get img tensor
            fake_img_ = fake_img_['data']
            fake_imgs.append(fake_img_)
        fake_imgs = torch.stack(fake_imgs, dim=0)
        fake_imgs = self._preprocess(fake_imgs).to(self.device)

        if self.inception_style == 'StyleGAN':
            fake_imgs = (fake_imgs * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            with disable_gpu_fuser_on_pt19():
                feat = self.inception(fake_imgs, no_output_bias=True)
        else:
            feat = F.softmax(self.inception(fake_imgs), dim=1)

        # NOTE: feat is shape like (bz, 1000), convert to a list
        self.fake_results += list(torch.split(feat, 1))

    def get_metric_sampler(self, model: nn.Module, dataloader: DataLoader,
                           metrics: List['GenMetric']) -> DataLoader:
        """Get sampler for normal metrics. Directly returns the dataloader.

        Args:
            model (nn.Module): Model to evaluate.
            dataloader (DataLoader): Dataloader for real images.
            metrics (List['GenMetric']): Metrics with the same sample mode.

        Returns:
            DataLoader: Default sampler for normal metrics.
        """
        return dataloader


@METRICS.register_module('PPL')
@METRICS.register_module()
class PerceptualPathLength(GenerativeMetric):
    r"""Perceptual path length.

        Measure the difference between consecutive images (their VGG16
        embeddings) when interpolating between two random inputs. Drastic
        changes mean that multiple features have changed together and that
        they might be entangled.

        Ref: https://github.com/rosinality/stylegan2-pytorch/blob/master/ppl.py # noqa

        Args:
            num_images (int): The number of evaluated generated samples.
            image_shape (tuple, optional): Image shape in order "CHW". Defaults
                to None.
            crop (bool, optional): Whether crop images. Defaults to True.
            epsilon (float, optional): Epsilon parameter for path sampling.
                Defaults to 1e-4.
            space (str, optional): Latent space. Defaults to 'W'.
            sampling (str, optional): Sampling mode, whether sampling in full
                path or endpoints. Defaults to 'end'.
            latent_dim (int, optional): Latent dimension of input noise.
                Defaults to 512.
    """
    SAMPLER_MODE = 'path'

    def __init__(self,
                 fake_nums: int,
                 real_nums: int = 0,
                 fake_key: Optional[str] = None,
                 real_key: Optional[str] = 'img',
                 sample_model: str = 'ema',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 crop=True,
                 epsilon=1e-4,
                 space='W',
                 sampling='end',
                 latent_dim=512):
        super().__init__(fake_nums, real_nums, fake_key, real_key,
                         sample_model, collect_device, prefix)
        self.crop = crop

        self.epsilon = epsilon
        self.space = space
        self.sampling = sampling
        self.latent_dim = latent_dim

    @torch.no_grad()
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.fake_results``, which will be used
        to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        fake_imgs = []
        for pred in data_samples:
            fake_img_ = pred
            # get ema/orig results
            if self.sample_model in fake_img_:
                fake_img_ = fake_img_[self.sample_model]
            # get specific fake_keys
            if (self.fake_key is not None and self.fake_key in fake_img_):
                fake_img_ = fake_img_[self.fake_key]['data']
            else:
                # get img tensor
                fake_img_ = fake_img_['fake_img']['data']
            fake_imgs.append(fake_img_)
        fake_imgs = torch.stack(fake_imgs, dim=0)
        feat = self._compute_distance(fake_imgs)
        feat_list = list(torch.split(feat, 1))
        self.fake_results += feat_list

    @torch.no_grad()
    def _compute_distance(self, images):
        """Feed data to the metric.

        Args:
            images (Tensor): Input tensor.
        """
        # use minibatch's device type to initialize a lpips calculator
        if not hasattr(self, 'percept'):
            self.percept = PerceptualLoss(
                use_gpu=images.device.type.startswith('cuda'))
        # crop and resize images
        if self.crop:
            c = images.shape[2] // 8
            minibatch = images[:, :, c * 3:c * 7, c * 2:c * 6]

        factor = minibatch.shape[2] // 256
        if factor > 1:
            minibatch = F.interpolate(
                minibatch,
                size=(256, 256),
                mode='bilinear',
                align_corners=False)
        # calculator and store lpips score
        distance = self.percept(minibatch[::2], minibatch[1::2]).view(
            minibatch.shape[0] // 2) / (
                self.epsilon**2)
        return distance.to('cpu')

    @torch.no_grad()
    def compute_metrics(self, fake_results: list) -> dict:
        """Summarize the results.

        Returns:
            dict | list: Summarized results.
        """
        distances = torch.cat(self.fake_results, dim=0).numpy()
        lo = np.percentile(distances, 1, interpolation='lower')
        hi = np.percentile(distances, 99, interpolation='higher')
        filtered_dist = np.extract(
            np.logical_and(lo <= distances, distances <= hi), distances)
        ppl_score = float(filtered_dist.mean())
        return {'ppl_score': ppl_score}

    def get_metric_sampler(self, model: nn.Module, dataloader: DataLoader,
                           metrics: GenMetric):
        """Get sampler for generative metrics. Returns a dummy iterator, whose
        return value of each iteration is a dict containing batch size and
        sample mode to generate images.

        Args:
            model (nn.Module): Model to evaluate.
            dataloader (DataLoader): Dataloader for real images. Used to get
                batch size during generate fake images.
            metrics (List['GenMetric']): Metrics with the same sampler mode.

        Returns:
            :class:`dummy_iterator`: Sampler for generative metrics.
        """

        batch_size = dataloader.batch_size

        sample_model = metrics[0].sample_model
        assert all([metric.sample_model == sample_model for metric in metrics
                    ]), ('\'sample_model\' between metrics is inconsistency.')

        class PPLSampler:
            """StyleGAN series generator's sampling iterator for PPL metric.

            Args:
                generator (nn.Module): StyleGAN series' generator.
                num_images (int): The number of evaluated generated samples.
                batch_size (int): Batch size of generated images.
                space (str, optional): Latent space. Defaults to 'W'.
                sampling (str, optional): Sampling mode, whether sampling in
                    full path or endpoints. Defaults to 'end'.
                epsilon (float, optional): Epsilon parameter for path sampling.
                    Defaults to 1e-4.
                latent_dim (int, optional): Latent dimension of input noise.
                    Defaults to 512.
            """

            def __init__(self,
                         generator,
                         num_images,
                         batch_size,
                         space='W',
                         sampling='end',
                         epsilon=1e-4,
                         latent_dim=512):
                assert space in ['Z', 'W']
                assert sampling in ['full', 'end']
                n_batch = num_images // batch_size

                resid = num_images - (n_batch * batch_size)
                self.batch_sizes = [batch_size] * n_batch + ([resid] if
                                                             resid > 0 else [])
                self.device = get_module_device(generator)
                self.generator = generator.module if hasattr(
                    generator, 'module') else generator
                self.latent_dim = latent_dim
                self.space = space
                self.sampling = sampling
                self.epsilon = epsilon

            def __iter__(self):
                self.idx = 0
                return self

            def __len__(self):
                return len(self.batch_sizes)

            @torch.no_grad()
            def __next__(self):
                if self.idx >= len(self.batch_sizes):
                    raise StopIteration
                batch = self.batch_sizes[self.idx]
                inputs = torch.randn([batch * 2, self.latent_dim],
                                     device=self.device)
                if self.sampling == 'full':
                    lerp_t = torch.rand(batch, device=self.device)
                else:
                    lerp_t = torch.zeros(batch, device=self.device)

                if self.space == 'W':
                    assert hasattr(self.generator, 'style_mapping')
                    latent = self.generator.style_mapping(inputs)
                    latent_t0, latent_t1 = latent[::2], latent[1::2]
                    latent_e0 = torch.lerp(latent_t0, latent_t1, lerp_t[:,
                                                                        None])
                    latent_e1 = torch.lerp(latent_t0, latent_t1,
                                           lerp_t[:, None] + self.epsilon)
                    latent_e = torch.stack([latent_e0, latent_e1],
                                           1).view(*latent.shape)
                else:
                    latent_t0, latent_t1 = inputs[::2], inputs[1::2]
                    latent_e0 = slerp(latent_t0, latent_t1, lerp_t[:, None])
                    latent_e1 = slerp(latent_t0, latent_t1,
                                      lerp_t[:, None] + self.epsilon)
                    latent_e = torch.stack([latent_e0, latent_e1],
                                           1).view(*inputs.shape)

                self.idx += 1
                return dict(
                    inputs=dict(
                        noise=latent_e,
                        sample_kwargs=dict(
                            input_is_latent=(self.space == 'W'))))

        ppl_sampler = PPLSampler(
            model.generator_ema
            if self.sample_model == 'ema' else model.generator,
            num_images=max([metric.fake_nums_per_device
                            for metric in metrics]),
            batch_size=batch_size,
            space=self.space,
            sampling=self.sampling,
            epsilon=self.epsilon,
            latent_dim=self.latent_dim)
        return ppl_sampler
