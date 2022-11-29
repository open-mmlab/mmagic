# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence

import lpips
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

from mmedit.models.utils import get_module_device, normalize_vecs
from mmedit.registry import METRICS
from .base_gen_metric import GenerativeMetric


def slerp(a, b, percent):
    """Spherical linear interpolation between two unnormalized vectors.

    Args:
        a (Tensor): Tensor with shape [N, C].
        b (Tensor): Tensor with shape [N, C].
        percent (float|Tensor): A float or tensor with shape broadcastable to
            the shape of input Tensors.

    Returns:
        Tensor: Spherical linear interpolation result with shape [N, C].
    """
    a = normalize_vecs(a)
    b = normalize_vecs(b)
    d = (a * b).sum(-1, keepdim=True)
    p = percent * torch.acos(d)
    c = normalize_vecs(b - d * a)
    d = a * torch.cos(p) + c * torch.sin(p)

    return normalize_vecs(d)


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
            need_cond_input (bool): If true, the sampler will return the
                conditional input randomly sampled from the original dataset.
                This require the dataset implement `get_data_info` and field
                `gt_label` must be contained in the return value of
                `get_data_info`. Noted that, for unconditional models, set
                `need_cond_input` as True may influence the result of evaluation
                results since the conditional inputs are sampled from the dataset
                distribution; otherwise will be sampled from the uniform
                distribution. Defaults to False.
    """
    SAMPLER_MODE = 'path'

    def __init__(self,
                 fake_nums: int,
                 real_nums: int = 0,
                 fake_key: Optional[str] = None,
                 real_key: Optional[str] = 'img',
                 need_cond_input: bool = False,
                 sample_model: str = 'ema',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 crop=True,
                 epsilon=1e-4,
                 space='W',
                 sampling='end',
                 latent_dim=512):
        super().__init__(fake_nums, real_nums, fake_key, real_key,
                         need_cond_input, sample_model, collect_device, prefix)
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
            self.percept = lpips.LPIPS(net='vgg').to(images.device)
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
                           metrics: list):
        """Get sampler for generative metrics. Returns a dummy iterator, whose
        return value of each iteration is a dict containing batch size and
        sample mode to generate images.

        Args:
            model (nn.Module): Model to evaluate.
            dataloader (DataLoader): Dataloader for real images. Used to get
                batch size during generate fake images.
            metrics (list): Metrics with the same sampler mode.

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
                injected_noise = self.generator.make_injected_noise()
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
                            injected_noise=injected_noise,
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
