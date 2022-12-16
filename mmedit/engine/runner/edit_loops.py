# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, List, Sequence, Union

import torch
from mmengine.evaluator import BaseMetric, Evaluator
from mmengine.runner.amp import autocast
from mmengine.runner.base_loop import BaseLoop
from mmengine.utils import is_list_of
from torch.utils.data import DataLoader

from mmedit.registry import LOOPS

DATALOADER_TYPE = Union[DataLoader, Dict, List]
EVALUATOR_TYPE = Union[Evaluator, Dict, List]


@LOOPS.register_module()
class EditValLoop(BaseLoop):

    def __init__(self, runner, dataloader, evaluator, fp16=False):
        self._runner = runner

        self.dataloaders = self._build_dataloaders(dataloader)
        self.evaluators = self._build_evaluators(evaluator)

        self.fp16 = fp16

        assert len(self.dataloaders) == len(self.evaluators), (
            'Length of dataloaders and evaluators must be same, but receive '
            f'\'{len(self.dataloaders)}\' and \'{len(self.evaluators)}\''
            'respectively.')

    def _build_dataloaders(self,
                           dataloader: DATALOADER_TYPE) -> List[DataLoader]:
        runner = self._runner

        if not isinstance(dataloader, list):
            dataloader = [dataloader]

        dataloaders = []
        for loader in dataloader:
            if isinstance(loader, dict):
                dataloaders.append(
                    runner.build_dataloader(loader, seed=runner.seed))
            else:
                dataloaders.append(loader)

        return dataloaders

    def _build_evaluators(self, evaluator: EVALUATOR_TYPE) -> List[Evaluator]:
        runner = self._runner

        # evaluator: [dict, dict, dict], dict, [[dict], [dict]]
        # -> [[dict, dict, dict]], [dict], ...
        if not is_list_of(evaluator, list):
            evaluator = [evaluator]

        evaluators = [runner.build_evaluator(eval) for eval in evaluator]

        return evaluators

    def run(self):

        self._runner.call_hook('before_val')
        self._runner.call_hook('before_val_epoch')
        self._runner.model.eval()

        # access to the true model
        module = self._runner.model
        if hasattr(self.runner.model, 'module'):
            module = module.module

        multi_metric = dict()
        idx_counter = 0
        self.total_length = 0

        # 1. prepare all metrics and get the total length
        metrics_sampler_lists = []
        meta_info_list = []
        dataset_name_list = []
        for evaluator, dataloader in zip(self.evaluators, self.dataloaders):
            # 1.1 prepare for metrics
            evaluator.prepare_metrics(module, dataloader)
            # 1.2 prepare for metric-sampler pair
            metrics_sampler_list = evaluator.prepare_samplers(
                module, dataloader)
            metrics_sampler_lists.append(metrics_sampler_list)
            # 1.3 update total length
            self.total_length += sum([
                len(metrics_sampler[1])
                for metrics_sampler in metrics_sampler_list
            ])
            # 1.4 save metainfo and dataset's name
            meta_info_list.append(
                getattr(dataloader.dataset, 'metainfo', None))
            dataset_name_list.append(
                self.dataloader.dataset.__class__.__name__)

        # 2. run evaluation
        for idx in range(len(self.evaluators)):
            # 2.1 set self.evaluator for run_iter
            self.evaluator = self.evaluators[idx]

            # 2.2 update metainfo for evaluator and visualizer
            meta_info = meta_info_list[idx]
            dataset_name = dataset_name_list[idx]
            if meta_info:
                self.evaluator.dataset_meta = self.dataloader.dataset.metainfo
                self._runner.visualizer.dataset_meta = \
                    self.dataloader.dataset.metainfo
            else:
                warnings.warn(
                    f'Dataset {dataset_name} has no metainfo. `dataset_meta` '
                    'in evaluator, metric and visualizer will be None.')

            # 2.3 generate images
            metrics_sampler_list = metrics_sampler_lists[idx]
            for metrics, sampler in metrics_sampler_list:
                for data in sampler:
                    self.run_iter(idx_counter, data, metrics)
                    idx_counter += 1

            # 2.4 evaluate metrics and update multi_metric
            metrics = self.evaluator.evaluate()
            if multi_metric and metrics.keys() & multi_metric.keys():
                raise ValueError('Please set different prefix for different'
                                 ' datasets in `val_evaluator`')
            else:
                multi_metric.update(metrics)

        # 3. finish evaluation and call hooks
        self._runner.call_hook('after_val_epoch', metrics=multi_metric)
        self._runner.call_hook('after_val')

    @torch.no_grad()
    def run_iter(self, idx, data_batch: dict, metrics: Sequence[BaseMetric]):
        """Iterate one mini-batch and feed the output to corresponding
        `metrics`.

        Args:
            idx (int): Current idx for the input data.
            data_batch (dict): Batch of data from dataloader.
            metrics (Sequence[BaseMetric]): Specific metrics to evaluate.
        """
        self._runner.call_hook(
            'before_val_iter', batch_idx=idx, data_batch=data_batch)
        # outputs should be sequence of BaseDataElement
        with autocast(enabled=self.fp16):
            outputs = self._runner.model.val_step(data_batch)
        self.evaluator.process(outputs, data_batch, metrics)
        self._runner.call_hook(
            'after_val_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)


@LOOPS.register_module()
class EditTestLoop(BaseLoop):

    def __init__(self, runner, dataloader, evaluator, fp16=False):
        self._runner = runner

        self.dataloaders = self._build_dataloaders(dataloader)
        self.evaluators = self._build_evaluators(evaluator)

        self.fp16 = fp16

        assert len(self.dataloaders) == len(self.evaluators), (
            'Length of dataloaders and evaluators must be same, but receive '
            f'\'{len(self.dataloaders)}\' and \'{len(self.evaluators)}\''
            'respectively.')

    def _build_dataloaders(self,
                           dataloader: DATALOADER_TYPE) -> List[DataLoader]:

        runner = self._runner

        if not isinstance(dataloader, list):
            dataloader = [dataloader]

        dataloaders = []
        for loader in dataloader:
            if isinstance(loader, dict):
                dataloaders.append(
                    runner.build_dataloader(loader, seed=runner.seed))
            else:
                dataloaders.append(loader)

        return dataloaders

    def _build_evaluators(self, evaluator: EVALUATOR_TYPE) -> List[Evaluator]:
        runner = self._runner

        def is_evaluator_cfg(cfg):
            # Single evaluator with type
            if isinstance(cfg, dict) and 'metrics' in cfg:
                return True
            # Single evaluator without type
            elif (is_list_of(cfg, dict)
                  and all(['metrics' not in cfg_ for cfg_ in cfg])):
                return True
            else:
                return False

        # Input type checking and packing
        # 1. Single evaluator without type: [dict(), dict(), ...]
        # 2. Single evaluator with type: dict(type=xx, metrics=xx)
        # 3. Multi evaluator without type: [[dict, ...], [dict, ...]]
        # 4. Multi evaluator with type: [dict(type=xx, metrics=xx), dict(...)]
        if is_evaluator_cfg(evaluator):
            evaluator = [evaluator]
        else:
            assert all([
                is_evaluator_cfg(cfg) for cfg in evaluator
            ]), ('Unsupport evaluator type, please check your input and '
                 'the docstring.')

        evaluators = [runner.build_evaluator(eval) for eval in evaluator]

        return evaluators

    def run(self):

        self._runner.call_hook('before_test')
        self._runner.call_hook('before_test_epoch')
        self._runner.model.eval()

        # access to the true model
        module = self._runner.model
        if hasattr(self._runner.model, 'module'):
            module = module.module

        multi_metric = dict()
        idx_counter = 0
        self.total_length = 0

        # 1. prepare all metrics and get the total length
        metrics_sampler_lists = []
        meta_info_list = []
        dataset_name_list = []
        for evaluator, dataloader in zip(self.evaluators, self.dataloaders):
            # 1.1 prepare for metrics
            evaluator.prepare_metrics(module, dataloader)
            # 1.2 prepare for metric-sampler pair
            metrics_sampler_list = evaluator.prepare_samplers(
                module, dataloader)
            metrics_sampler_lists.append(metrics_sampler_list)
            # 1.3 update total length
            self.total_length += sum([
                len(metrics_sampler[1])
                for metrics_sampler in metrics_sampler_list
            ])
            # 1.4 save metainfo and dataset's name
            meta_info_list.append(
                getattr(dataloader.dataset, 'metainfo', None))
            dataset_name_list.append(dataloader.dataset.__class__.__name__)

        # 2. run evaluation
        for idx in range(len(self.evaluators)):
            # 2.1 set self.evaluator for run_iter
            self.evaluator = self.evaluators[idx]
            self.dataloader = self.dataloaders[idx]

            # 2.2 update metainfo for evaluator and visualizer
            meta_info = meta_info_list[idx]
            dataset_name = dataset_name_list[idx]
            if meta_info:
                self.evaluator.dataset_meta = meta_info
                self._runner.visualizer.dataset_meta = meta_info
            else:
                warnings.warn(
                    f'Dataset {dataset_name} has no metainfo. `dataset_meta` '
                    'in evaluator, metric and visualizer will be None.')

            # 2.3 generate images
            metrics_sampler_list = metrics_sampler_lists[idx]
            for metrics, sampler in metrics_sampler_list:
                for data in sampler:
                    self.run_iter(idx_counter, data, metrics)
                    idx_counter += 1

            # 2.4 evaluate metrics and update multi_metric
            metrics = self.evaluator.evaluate()
            if multi_metric and metrics.keys() & multi_metric.keys():
                raise ValueError('Please set different prefix for different'
                                 ' datasets in `test_evaluator`')
            else:
                multi_metric.update(metrics)

        # 3. finish evaluation and call hooks
        self._runner.call_hook('after_test_epoch', metrics=multi_metric)
        self._runner.call_hook('after_test')

    @torch.no_grad()
    def run_iter(self, idx, data_batch: dict, metrics: Sequence[BaseMetric]):
        """Iterate one mini-batch and feed the output to corresponding
        `metrics`.

        Args:
            idx (int): Current idx for the input data.
            data_batch (dict): Batch of data from dataloader.
            metrics (Sequence[BaseMetric]): Specific metrics to evaluate.
        """
        self._runner.call_hook(
            'before_test_iter', batch_idx=idx, data_batch=data_batch)
        # outputs should be sequence of BaseDataElement
        with autocast(enabled=self.fp16):
            outputs = self._runner.model.test_step(data_batch)
        self.evaluator.process(outputs, data_batch, metrics)
        self._runner.call_hook(
            'after_test_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
