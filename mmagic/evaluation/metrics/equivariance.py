# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict
from copy import deepcopy
from typing import Iterator, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from mmengine.dist import all_gather
from torch.utils.data.dataloader import DataLoader

from mmagic.registry import METRICS
from .base_gen_metric import GenerativeMetric


@METRICS.register_module('EQ')
@METRICS.register_module()
class Equivariance(GenerativeMetric):

    name = 'Equivariance'

    def __init__(self,
                 fake_nums: int,
                 real_nums: int = 0,
                 fake_key: Optional[str] = None,
                 real_key: Optional[str] = 'gt_img',
                 need_cond_input: bool = False,
                 sample_mode: str = 'ema',
                 sample_kwargs: dict = dict(),
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 eq_cfg=dict()):
        super().__init__(fake_nums, real_nums, fake_key, real_key,
                         need_cond_input, sample_mode, collect_device, prefix)
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
                           metrics: List[GenerativeMetric]):
        """Get sampler for generative metrics. Returns a dummy iterator, whose
        return value of each iteration is a dict containing batch size and
        sample mode to generate images.

        Args:
            model (nn.Module): Model to evaluate.
            dataloader (DataLoader): Dataloader for real images. Used to get
                batch size during generate fake images.
            metrics (List['GenerativeMetric']): Metrics with the same sampler
                mode.

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
