# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, List, Sequence, Union

import torch
from mmengine.evaluator import Evaluator
from mmengine.runner.amp import autocast
from mmengine.runner.base_loop import BaseLoop
from mmengine.utils import is_list_of
from torch.utils.data import DataLoader

from mmedit.registry import LOOPS


@LOOPS.register_module()
class MultiValLoop(BaseLoop):
    """Loop for validation multi-datasets.

    Args:
        runner (Runner): A reference of runner.
        dataloader (list[Dataloader or dic]): A dataloader object or a dict to
            build a dataloader.
        evaluator (list[]): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 validation. Defaults to False.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 fp16: bool = False) -> None:
        self._runner = runner
        assert isinstance(dataloader, list)
        self.dataloaders = list()
        for loader in dataloader:
            if isinstance(loader, dict):
                self.dataloaders.append(
                    runner.build_dataloader(loader, seed=runner.seed))
            else:
                self.dataloaders.append(loader)

        assert isinstance(evaluator, list)
        self.evaluators = list()
        for single_evalator in evaluator:
            if isinstance(single_evalator, dict) or is_list_of(
                    single_evalator, dict):
                self.evaluators.append(runner.build_evaluator(single_evalator))
            else:
                self.evaluators.append(single_evalator)
        self.evaluators = [runner.build_evaluator(eval) for eval in evaluator]

        assert len(self.evaluators) == len(self.dataloaders)

        self.fp16 = fp16

    def run(self):
        """Launch validation."""
        self.runner.call_hook('before_val')

        self.runner.model.eval()
        multi_metric = dict()
        self.runner.call_hook('before_val_epoch')
        for evaluator, dataloader in zip(self.evaluators, self.dataloaders):
            self.evaluator = evaluator
            self.dataloader = dataloader
            if hasattr(self.dataloader.dataset, 'metainfo'):
                self.evaluator.dataset_meta = self.dataloader.dataset.metainfo
                self.runner.visualizer.dataset_meta = \
                    self.dataloader.dataset.metainfo
            else:
                warnings.warn(
                    f'Dataset {self.dataloader.dataset.__class__.__name__} '
                    'has no metainfo. ``dataset_meta`` in evaluator, metric'
                    ' and visualizer will be None.')
            for idx, data_batch in enumerate(self.dataloader):
                self.run_iter(idx, data_batch)
                # compute metrics
            metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
            if multi_metric and metrics.keys() & multi_metric.keys():
                raise ValueError('Please set different prefix for different'
                                 ' datasets in `val_evaluator`')
            else:

                multi_metric.update(metrics)
        self.runner.call_hook('after_val_epoch', metrics=multi_metric)
        self.runner.call_hook('after_val')

    @torch.no_grad()
    def run_iter(self, idx: int, data_batch: Sequence[dict]):
        """Iterate one mini-batch.

        Args:
            idx (int): The index of the current batch in the loop.
            data_batch (Sequence[dict]): Batch of data
                from dataloader.
        """
        self.runner.call_hook(
            'before_val_iter', batch_idx=idx, data_batch=data_batch)
        # outputs should be sequence of BaseDataElement
        with autocast(enabled=self.fp16):
            outputs = self.runner.model.val_step(data_batch)
        self.evaluator.process(outputs, data_batch)
        self.runner.call_hook(
            'after_val_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)


@LOOPS.register_module()
class MultiTestLoop(BaseLoop):
    """Loop for validation multi-datasets.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 validation. Defaults to False.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 fp16: bool = False) -> None:
        self._runner = runner
        assert isinstance(dataloader, list)
        self.dataloaders = list()
        for loader in dataloader:
            if isinstance(loader, dict):
                self.dataloaders.append(
                    runner.build_dataloader(loader, seed=runner.seed))
            else:
                self.dataloaders.append(loader)

        assert isinstance(evaluator, list)
        self.evaluators = list()
        for single_evalator in evaluator:
            if isinstance(single_evalator, dict) or is_list_of(
                    single_evalator, dict):
                self.evaluators.append(runner.build_evaluator(single_evalator))
            else:
                self.evaluators.append(single_evalator)
        self.evaluators = [runner.build_evaluator(eval) for eval in evaluator]

        assert len(self.evaluators) == len(self.dataloaders)

        self.fp16 = fp16

    def run(self):
        """Launch test."""
        self.runner.call_hook('before_test')

        self.runner.model.eval()
        multi_metric = dict()
        self.runner.call_hook('before_test_epoch')
        for evaluator, dataloader in zip(self.evaluators, self.dataloaders):
            self.dataloader = dataloader
            self.evaluator = evaluator
            if hasattr(self.dataloader.dataset, 'metainfo'):
                self.evaluator.dataset_meta = self.dataloader.dataset.metainfo
                self.runner.visualizer.dataset_meta = \
                    self.dataloader.dataset.metainfo
            else:
                warnings.warn(
                    f'Dataset {self.dataloader.dataset.__class__.__name__} '
                    'has no metainfo. ``dataset_meta`` in evaluator, metric'
                    ' and visualizer will be None.')
            for idx, data_batch in enumerate(self.dataloader):
                self.run_iter(idx, data_batch)
                # compute metrics
            metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
            if multi_metric and metrics.keys() & multi_metric.keys():
                raise ValueError('Please set different prefix for different'
                                 ' datasets in `test_evaluator`')
            else:

                multi_metric.update(metrics)
        self.runner.call_hook('after_test_epoch', metrics=multi_metric)
        self.runner.call_hook('after_test')

    @torch.no_grad()
    def run_iter(self, idx: int, data_batch: Sequence[dict]):
        """Iterate one mini-batch.

        Args:
            idx (int): The index of the current batch in the loop.
            data_batch (Sequence[dict]): Batch of data
                from dataloader.
        """
        self.runner.call_hook(
            'before_test_iter', batch_idx=idx, data_batch=data_batch)
        # outputs should be sequence of BaseDataElement
        with autocast(enabled=self.fp16):
            predictions = self.runner.model.test_step(data_batch)
        self.evaluator.process(predictions, data_batch)
        self.runner.call_hook(
            'after_test_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=predictions)
