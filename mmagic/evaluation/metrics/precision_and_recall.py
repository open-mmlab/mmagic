# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine import print_log
from torch.utils.data.dataloader import DataLoader
from torchvision import models as torchvision_models

from mmagic.models.utils import get_module_device
from mmagic.registry import METRICS
from ..functional import prepare_vgg_feat
from .base_gen_metric import GenerativeMetric


def compute_pr_distances(row_features,
                         col_features,
                         num_gpus=1,
                         rank=0,
                         col_batch_size=10000):
    r"""Compute distances between real images and fake images.

    This function is used for calculate Precision and Recall metric.
    Refer to:https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/metrics/precision_recall.py  # noqa
    """
    assert 0 <= rank < num_gpus
    num_cols = col_features.shape[0]
    num_batches = ((num_cols - 1) // col_batch_size // num_gpus + 1) * num_gpus
    col_batches = torch.nn.functional.pad(col_features,
                                          [0, 0, 0, -num_cols % num_batches
                                           ]).chunk(num_batches)
    dist_batches = []
    for col_batch in col_batches[rank::num_gpus]:
        dist_batch = torch.cdist(
            row_features.unsqueeze(0), col_batch.unsqueeze(0))[0]
        for src in range(num_gpus):
            dist_broadcast = dist_batch.clone()
            if num_gpus > 1:
                torch.distributed.broadcast(dist_broadcast, src=src)
            dist_batches.append(dist_broadcast.cpu() if rank == 0 else None)
    return torch.cat(dist_batches, dim=1)[:, :num_cols] if rank == 0 else None


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
    name = 'PR'

    def __init__(self,
                 fake_nums,
                 real_nums=-1,
                 k=3,
                 fake_key: Optional[str] = None,
                 real_key: Optional[str] = 'gt_img',
                 need_cond_input: bool = False,
                 sample_model: str = 'ema',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 vgg16_script='work_dirs/cache/vgg16.pt',
                 vgg16_pkl=None,
                 row_batch_size=10000,
                 col_batch_size=10000,
                 auto_save=True):
        super().__init__(fake_nums, real_nums, fake_key, real_key,
                         need_cond_input, sample_model, collect_device, prefix)
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
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extracting image features.

        Args:
            images (torch.Tensor): Images tensor.
        Returns:
            torch.Tensor: Vgg16 features of input images.
        """
        # image must passed in 'bgr'
        images = images[:, [2, 1, 0], ...]
        if self.use_tero_scirpt:
            images = images.to(torch.uint8)
            feature = self.vgg16(images, return_features=True)
        else:
            images = (images - 127.5) / 127.5
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
                fake_img_ = fake_img_[self.fake_key]
            else:
                # get img tensor
                fake_img_ = fake_img_['fake_img']
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
