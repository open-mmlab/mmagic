# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, List, Sequence, Union

import torch
from mmengine.evaluator import BaseMetric, Evaluator
from mmengine.runner.amp import autocast
from mmengine.runner.base_loop import BaseLoop
from torch.utils.data import DataLoader

from mmagic.registry import LOOPS
from .loop_utils import is_evaluator, update_and_check_evaluator

DATALOADER_TYPE = Union[DataLoader, Dict, List]
EVALUATOR_TYPE = Union[Evaluator, Dict, List]


@LOOPS.register_module()
class MultiValLoop(BaseLoop):
    """Validation loop for MMagic models which support evaluate multiply
    dataset at the same time. This class support evaluate:

    1. Metrics (metric) on a single dataset (e.g. PSNR and SSIM on DIV2K
       dataset)
    2. Different metrics on different datasets (e.g. PSNR on DIV2K and SSIM
       and PSNR on SET5)

    Use cases:

    Case 1: metrics on a single dataset

    >>> # add the following lines in your config
    >>> # 1. use `MultiValLoop` instead of `ValLoop` in MMEngine
    >>> val_cfg = dict(type='MultiValLoop')
    >>> # 2. specific MultiEvaluator instead of Evaluator in MMEngine
    >>> val_evaluator = dict(
    >>>     type='MultiEvaluator',
    >>>     metrics=[
    >>>         dict(type='PSNR', crop_border=2, prefix='Set5'),
    >>>         dict(type='SSIM', crop_border=2, prefix='Set5'),
    >>>     ])
    >>> # 3. define dataloader
    >>> val_dataloader = dict(...)

    Case 2: different metrics on different datasets

    >>> # add the following lines in your config
    >>> # 1. use `MultiValLoop` instead of `ValLoop` in MMEngine
    >>> val_cfg = dict(type='MultiValLoop')
    >>> # 2. specific a list MultiEvaluator
    >>> # do not forget to add prefix for each metric group
    >>> div2k_evaluator = dict(
    >>>     type='MultiEvaluator',
    >>>     metrics=dict(type='SSIM', crop_border=2, prefix='DIV2K'))
    >>> set5_evaluator = dict(
    >>>     type='MultiEvaluator',
    >>>     metrics=[
    >>>         dict(type='PSNR', crop_border=2, prefix='Set5'),
    >>>         dict(type='SSIM', crop_border=2, prefix='Set5'),
    >>>     ])
    >>> # define evaluator config
    >>> val_evaluator = [div2k_evaluator, set5_evaluator]
    >>> # 3. specific a list dataloader for each metric groups
    >>> div2k_dataloader = dict(...)
    >>> set5_dataloader = dict(...)
    >>> # define dataloader config
    >>> val_dataloader = [div2k_dataloader, set5_dataloader]

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict or list): A dataloader object or a dict
            to build a dataloader a list of dataloader object or a list of
            config dicts.
        evaluator (Evaluator or dict or list): A evaluator object or a dict to
            build the evaluator or a list of evaluator object or a list of
            config dicts.
    """

    def __init__(self,
                 runner,
                 dataloader: DATALOADER_TYPE,
                 evaluator: EVALUATOR_TYPE,
                 fp16: bool = False):
        self._runner = runner

        self.dataloaders = self._build_dataloaders(dataloader)
        self.evaluators = self._build_evaluators(evaluator)

        self.fp16 = fp16

        assert len(self.dataloaders) == len(self.evaluators), (
            'Length of dataloaders and evaluators must be same, but receive '
            f'\'{len(self.dataloaders)}\' and \'{len(self.evaluators)}\''
            'respectively.')

        self._total_length = None  # length for all dataloaders

    @property
    def total_length(self) -> int:
        if self._total_length is not None:
            return self._total_length

        warnings.warn('\'total_length\' has not been initialized and return '
                      '\'0\' for safety. This result is likely to be incorrect'
                      ' and we recommend you to call \'total_length\' after '
                      '\'self.run\' is called.')
        return 0

    def _build_dataloaders(self,
                           dataloader: DATALOADER_TYPE) -> List[DataLoader]:
        """Build dataloaders.

        Args:
            dataloader (Dataloader or dict or list): A dataloader object or a
                dict to build a dataloader a list of dataloader object or a
                list of config dict.

        Returns:
            List[Dataloader]: List of dataloaders for compute metrics.
        """
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
        """Build evaluators.

        Args:
            evaluator (Evaluator or dict or list): A evaluator object or a
                dict to build the evaluator or a list of evaluator object or a
                list of config dicts.

        Returns:
            List[Evaluator]: List of evaluators for compute metrics.
        """
        runner = self._runner

        # Input type checking and packing
        # 1. Single evaluator without type: [dict(), dict(), ...]
        # 2. Single evaluator with type: dict(type=xx, metrics=xx)
        # 3. Multi evaluator without type: [[dict, ...], [dict, ...]]
        # 4. Multi evaluator with type: [dict(type=xx, metrics=xx), dict(...)]
        if is_evaluator(evaluator):
            evaluator = [update_and_check_evaluator(evaluator)]
        else:
            assert all([
                is_evaluator(cfg) for cfg in evaluator
            ]), ('Unsupported evaluator type, please check your input and '
                 'the docstring.')
            evaluator = [update_and_check_evaluator(cfg) for cfg in evaluator]

        evaluators = [runner.build_evaluator(eval) for eval in evaluator]

        return evaluators

    def run(self):
        """Launch validation. The evaluation process consists of four steps.

        1. Prepare pre-calculated items for all metrics by calling
           :meth:`self.evaluator.prepare_metrics`.
        2. Get a list of metrics-sampler pair. Each pair contains a list of
           metrics with the same sampler mode and a shared sampler.
        3. Generate images for the each metrics group. Loop for elements in
           each sampler and feed to the model as input by calling
           :meth:`self.run_iter`.
        4. Evaluate all metrics by calling :meth:`self.evaluator.evaluate`.
        """

        self._runner.call_hook('before_val')
        self._runner.call_hook('before_val_epoch')
        self._runner.model.eval()

        # access to the true model
        module = self._runner.model
        if hasattr(self.runner.model, 'module'):
            module = module.module

        multi_metric = dict()
        idx_counter = 0
        self._total_length = 0

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
            self._total_length += sum([
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
class MultiTestLoop(BaseLoop):
    """Test loop for MMagic models which support evaluate multiply dataset at
    the same time. This class support evaluate:

    1. Metrics (metric) on a single dataset (e.g. PSNR and SSIM on DIV2K
       dataset)
    2. Different metrics on different datasets (e.g. PSNR on DIV2K and SSIM
       and PSNR on SET5)

    Use cases:

    Case 1: metrics on a single dataset

    >>> # add the following lines in your config
    >>> # 1. use `MultiTestLoop` instead of `TestLoop` in MMEngine
    >>> val_cfg = dict(type='MultiTestLoop')
    >>> # 2. specific MultiEvaluator instead of Evaluator in MMEngine
    >>> test_evaluator = dict(
    >>>     type='MultiEvaluator',
    >>>     metrics=[
    >>>         dict(type='PSNR', crop_border=2, prefix='Set5'),
    >>>         dict(type='SSIM', crop_border=2, prefix='Set5'),
    >>>     ])
    >>> # 3. define dataloader
    >>> test_dataloader = dict(...)

    Case 2: different metrics on different datasets

    >>> # add the following lines in your config
    >>> # 1. use `MultiTestLoop` instead of `TestLoop` in MMEngine
    >>> Test_cfg = dict(type='MultiTestLoop')
    >>> # 2. specific a list MultiEvaluator
    >>> # do not forget to add prefix for each metric group
    >>> div2k_evaluator = dict(
    >>>     type='MultiEvaluator',
    >>>     metrics=dict(type='SSIM', crop_border=2, prefix='DIV2K'))
    >>> set5_evaluator = dict(
    >>>     type='MultiEvaluator',
    >>>     metrics=[
    >>>         dict(type='PSNR', crop_border=2, prefix='Set5'),
    >>>         dict(type='SSIM', crop_border=2, prefix='Set5'),
    >>>     ])
    >>> # define evaluator config
    >>> test_evaluator = [div2k_evaluator, set5_evaluator]
    >>> # 3. specific a list dataloader for each metric groups
    >>> div2k_dataloader = dict(...)
    >>> set5_dataloader = dict(...)
    >>> # define dataloader config
    >>> test_dataloader = [div2k_dataloader, set5_dataloader]

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict or list): A dataloader object or a dict
            to build a dataloader a list of dataloader object or a list of
            config dicts.
        evaluator (Evaluator or dict or list): A evaluator object or a dict to
            build the evaluator or a list of evaluator object or a list of
            config dicts.
    """

    def __init__(self, runner, dataloader, evaluator, fp16=False):
        self._runner = runner

        self.dataloaders = self._build_dataloaders(dataloader)
        self.evaluators = self._build_evaluators(evaluator)

        self.fp16 = fp16

        assert len(self.dataloaders) == len(self.evaluators), (
            'Length of dataloaders and evaluators must be same, but receive '
            f'\'{len(self.dataloaders)}\' and \'{len(self.evaluators)}\''
            'respectively.')

        self._total_length = None

    @property
    def total_length(self) -> int:
        if self._total_length is not None:
            return self._total_length

        warnings.warn('\'total_length\' has not been initialized and return '
                      '\'0\' for safety. This result is likely to be incorrect'
                      ' and we recommend you to call \'total_length\' after '
                      '\'self.run\' is called.')
        return 0

    def _build_dataloaders(self,
                           dataloader: DATALOADER_TYPE) -> List[DataLoader]:
        """Build dataloaders.

        Args:
            dataloader (Dataloader or dict or list): A dataloader object or a
                dict to build a dataloader a list of dataloader object or a
                list of config dict.

        Returns:
            List[Dataloader]: List of dataloaders for compute metrics.
        """
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
        """Build evaluators.

        Args:
            evaluator (Evaluator or dict or list): A evaluator object or a
                dict to build the evaluator or a list of evaluator object or a
                list of config dicts.

        Returns:
            List[Evaluator]: List of evaluators for compute metrics.
        """
        runner = self._runner

        # Input type checking and packing
        # 1. Single evaluator without type: [dict(), dict(), ...]
        # 2. Single evaluator with type: dict(type=xx, metrics=xx)
        # 3. Multi evaluator without type: [[dict, ...], [dict, ...]]
        # 4. Multi evaluator with type: [dict(type=xx, metrics=xx), dict(...)]
        if is_evaluator(evaluator):
            evaluator = [update_and_check_evaluator(evaluator)]
        else:
            assert all([
                is_evaluator(cfg) for cfg in evaluator
            ]), ('Unsupported evaluator type, please check your input and '
                 'the docstring.')
            evaluator = [update_and_check_evaluator(cfg) for cfg in evaluator]

        evaluators = [runner.build_evaluator(eval) for eval in evaluator]

        return evaluators

    def run(self):
        """Launch validation. The evaluation process consists of four steps.

        1. Prepare pre-calculated items for all metrics by calling
           :meth:`self.evaluator.prepare_metrics`.
        2. Get a list of metrics-sampler pair. Each pair contains a list of
           metrics with the same sampler mode and a shared sampler.
        3. Generate images for the each metrics group. Loop for elements in
           each sampler and feed to the model as input by calling
           :meth:`self.run_iter`.
        4. Evaluate all metrics by calling :meth:`self.evaluator.evaluate`.
        """

        self._runner.call_hook('before_test')
        self._runner.call_hook('before_test_epoch')
        self._runner.model.eval()

        # access to the true model
        module = self._runner.model
        if hasattr(self._runner.model, 'module'):
            module = module.module

        multi_metric = dict()
        idx_counter = 0
        self._total_length = 0

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
            self._total_length += sum([
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
