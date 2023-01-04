# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings
from typing import Any, Iterator, List, Optional

import numpy as np
import torch
import torch.nn as nn
from mmengine import is_list_of, print_log
from mmengine.dataset import pseudo_collate
from mmengine.dist import (all_gather, broadcast_object_list, collect_results,
                           get_dist_info, get_world_size, is_main_process)
from mmengine.evaluator import BaseMetric
from mmengine.model import is_model_wrapper
from torch import Tensor
from torch.utils.data.dataloader import DataLoader

from mmedit.structures import EditDataSample


class GenMetric(BaseMetric):
    """Metric for MMEditing.

    Args:
        fake_nums (int): Numbers of the generated image need for the metric.
        real_nums (int): Numbers of the real image need for the metric. If `-1`
            is passed means all images from the dataset is need. Defaults to 0.
        fake_key (Optional[str]): Key for get fake images of the output dict.
            Defaults to None.
        real_key (Optional[str]): Key for get real images from the input dict.
            Defaults to 'img'.
        sample_model (str): Sampling model for the generative model. Support
            'orig' and 'ema'. Defaults to 'ema'.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    SAMPLER_MODE = 'normal'

    def __init__(self,
                 fake_nums: int,
                 real_nums: int = 0,
                 fake_key: Optional[str] = None,
                 real_key: Optional[str] = 'img',
                 sample_model: str = 'ema',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device, prefix)
        self.sample_model = sample_model

        self.fake_nums = fake_nums
        self.real_nums = real_nums
        self.real_key = real_key
        self.fake_key = fake_key
        self.real_results: List[Any] = []
        self.fake_results: List[Any] = []

    @property
    def real_nums_per_device(self):
        """Number of real images need for current device."""
        return math.ceil(self.real_nums / get_world_size())

    @property
    def fake_nums_per_device(self):
        """Number of fake images need for current device."""
        return math.ceil(self.fake_nums / get_world_size())

    def _collect_target_results(self, target: str) -> Optional[list]:
        """Collected results in distributed environments.

        Args:
            target (str): Target results to collect.

        Returns:
            Optional[list]: The collected results.
        """
        assert target in [
            'fake', 'real'
        ], ('Only support to collect \'fake\' or \'real\' results.')
        results = getattr(self, f'{target}_results')
        size = getattr(self, f'{target}_nums')
        size = len(results) * get_world_size() if size == -1 else size

        if len(results) == 0:
            warnings.warn(
                f'{self.__class__.__name__} got empty `self.{target}_results`.'
                ' Please ensure that the processed results are properly added '
                f'into `self.{target}_results` in `process` method.')

        if is_list_of(results, Tensor):
            # apply all_gather for tensor results
            results = torch.cat(results, dim=0)
            results = torch.cat(all_gather(results), dim=0)[:size]
            results = torch.split(results, 1)
        else:
            # apply collect_results (all_gather_object) for non-tensor results
            results = collect_results(results, size, self.collect_device)

        # on non-main process, results should be `None`
        if is_main_process() and len(results) != size:
            raise ValueError(f'Length of results is \'{len(results)}\', not '
                             f'equals to target size \'{size}\'.')
        return results

    def evaluate(self) -> dict:
        """Evaluate the model performance of the whole dataset after processing
        all batches. Different like :class:`~mmengine.evaluator.BaseMetric`,
        this function evaluate the metric with paired results (`results_fake`
        and `results_real`).

        Returns:
            dict: Evaluation metrics dict on the val dataset. The keys are the
                names of the metrics, and the values are corresponding results.
        """

        results_fake = self._collect_target_results(target='fake')
        results_real = self._collect_target_results(target='real')

        if is_main_process():
            # pack to list, align with BaseMetrics
            _metrics = self.compute_metrics(results_fake, results_real)
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
        self.real_results.clear()
        self.fake_results.clear()

        return metrics[0]

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
        batch_size = dataloader.batch_size
        dataset_length = len(dataloader.dataset)
        rank, num_gpus = get_dist_info()
        assert self.real_nums <= dataset_length, (
            f'\'real_nums\'({self.real_nums}) can not larger than length of '
            f'dataset ({dataset_length}).')
        nums = dataset_length if self.real_nums == -1 else self.real_nums
        item_subset = [(i * num_gpus + rank) % nums
                       for i in range((nums - 1) // num_gpus + 1)]

        metric_dataloader = DataLoader(
            dataloader.dataset,
            batch_size=batch_size,
            sampler=item_subset,
            collate_fn=pseudo_collate,
            shuffle=False,
            drop_last=False)

        return metric_dataloader

    def compute_metrics(self, results_fake, results_real) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """

    def prepare(self, module: nn.Module, dataloader: DataLoader) -> None:
        """Prepare for the pre-calculating items of the metric. Defaults to do
        nothing.

        Args:
            module (nn.Module): Model to evaluate.
            dataloader (DataLoader): Dataloader for the real images.
        """
        if is_model_wrapper(module):
            module = module.module
        self.data_preprocessor = module.data_preprocessor


class GenerativeMetric(GenMetric):
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
        need_cond_input (bool): If true, the sampler will return the
            conditional input randomly sampled from the original dataset.
            This require the dataset implement `get_data_info` and field
            `gt_label` must be contained in the return value of
            `get_data_info`. Noted that, for unconditional models, set
            `need_cond_input` as True may influence the result of evaluation
            results since the conditional inputs are sampled from the dataset
            distribution; otherwise will be sampled from the uniform
            distribution. Defaults to False.
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
                 need_cond_input: bool = False,
                 sample_model: str = 'ema',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None):
        super().__init__(fake_nums, real_nums, fake_key, real_key,
                         sample_model, collect_device, prefix)
        self.need_cond_input = need_cond_input
        if self.need_cond_input:
            print_log('Set \'need_cond_input\' as True, this may influence '
                      'the evaluation results of conditional models.')

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
        dataset = dataloader.dataset

        sample_model = metrics[0].sample_model
        assert all([metric.sample_model == sample_model for metric in metrics
                    ]), ('\'sample_model\' between metrics is inconsistency.')

        class dummy_iterator:

            def __init__(self, batch_size, max_length, sample_model, dataset,
                         need_cond) -> None:
                self.batch_size = batch_size
                self.max_length = max_length
                self.sample_model = sample_model
                self.dataset = dataset
                self.need_cond = need_cond

            def __iter__(self) -> Iterator:
                self.idx = 0
                return self

            def __len__(self) -> int:
                return math.ceil(self.max_length / self.batch_size)

            def get_cond(self) -> List[EditDataSample]:

                data_sample_list = []
                for _ in range(self.batch_size):
                    data_sample = EditDataSample()
                    cond = self.dataset.get_data_info(
                        np.random.randint(len(self.dataset)))['gt_label']
                    data_sample.set_gt_label(torch.Tensor(cond))
                    data_sample_list.append(data_sample)
                return data_sample_list

            def __next__(self) -> dict:
                if self.idx > self.max_length:
                    raise StopIteration
                self.idx += batch_size

                output_dict = dict(
                    inputs=dict(
                        sample_model=self.sample_model,
                        num_batches=self.batch_size))

                if self.need_cond:
                    output_dict['data_samples'] = self.get_cond()

                return output_dict

        return dummy_iterator(
            batch_size=batch_size,
            max_length=max([metric.fake_nums_per_device
                            for metric in metrics]),
            sample_model=sample_model,
            dataset=dataset,
            need_cond=self.need_cond_input)

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
