# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Sequence, Union

import torch
from mmengine.evaluator import BaseMetric, Evaluator
from mmengine.registry import LOOPS
from mmengine.runner import Runner, TestLoop, ValLoop
from torch.utils.data import DataLoader


@LOOPS.register_module()
class GenValLoop(ValLoop):
    """Validation loop for generative models. This class support evaluate
    metrics with different sample mode.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
    """

    def __init__(self, runner: Runner, dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List]) -> None:

        super().__init__(runner, dataloader, evaluator)

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
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()

        # access to the true model
        module = self.runner.model
        if hasattr(self.runner.model, 'module'):
            module = module.module

        # 1. prepare for metrics
        self.evaluator.prepare_metrics(module, self.dataloader)

        # 2. prepare for metric-sampler pair
        metrics_sampler_list = self.evaluator.prepare_samplers(
            module, self.dataloader)
        # used for log processor
        self.total_length = sum([
            len(metrics_sampler[1]) for metrics_sampler in metrics_sampler_list
        ])

        # 3. generate images
        idx_counter = 0
        for metrics, sampler in metrics_sampler_list:
            for data in sampler:
                self.run_iter(idx_counter, data, metrics)
                idx_counter += 1

        # 4. evaluate metrics
        metrics = self.evaluator.evaluate()
        self.runner.call_hook('after_val_epoch', metrics=metrics)
        self.runner.call_hook('after_val')

    @torch.no_grad()
    def run_iter(self, idx, data_batch: dict, metrics: Sequence[BaseMetric]):
        """Iterate one mini-batch and feed the output to corresponding
        `metrics`.

        Args:
            idx (int): Current idx for the input data.
            data_batch (dict): Batch of data from dataloader.
            metrics (Sequence[BaseMetric]): Specific metrics to evaluate.
        """
        self.runner.call_hook(
            'before_val_iter', batch_idx=idx, data_batch=data_batch)
        # outputs should be sequence of BaseDataElement
        outputs = self.runner.model.val_step(data_batch)
        self.evaluator.process(outputs, data_batch, metrics)
        self.runner.call_hook(
            'after_val_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)


@LOOPS.register_module()
class GenTestLoop(TestLoop):
    """Validation loop for generative models. This class support evaluate
    metrics with different sample mode.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
    """

    def __init__(self, runner: Runner, dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List]) -> None:

        super().__init__(runner, dataloader, evaluator)

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
        self.runner.call_hook('before_test')
        self.runner.call_hook('before_test_epoch')
        self.runner.model.eval()

        # access to the true model
        module = self.runner.model
        if hasattr(self.runner.model, 'module'):
            module = module.module

        # 1. prepare for metrics
        self.evaluator.prepare_metrics(module, self.dataloader)

        # 2. prepare for metric-sampler pair
        metrics_sampler_list = self.evaluator.prepare_samplers(
            module, self.dataloader)
        # used for log processor
        self.total_length = sum([
            len(metrics_sampler[1]) for metrics_sampler in metrics_sampler_list
        ])

        idx_counter = 0
        for metrics, sampler in metrics_sampler_list:
            for data in sampler:
                self.run_iter(idx_counter, data, metrics)
                idx_counter += 1

        # 3. evaluate metrics
        metrics_output = self.evaluator.evaluate()
        self.runner.call_hook('after_test_epoch', metrics=metrics_output)
        self.runner.call_hook('after_test')

    @torch.no_grad()
    def run_iter(self, idx, data_batch: dict, metrics: Sequence[BaseMetric]):
        """Iterate one mini-batch and feed the output to corresponding
        `metrics`.

        Args:
            idx (int): Current idx for the input data.
            data_batch (dict): Batch of data from dataloader.
            metrics (Sequence[BaseMetric]): Specific metrics to evaluate.
        """
        self.runner.call_hook(
            'before_test_iter', batch_idx=idx, data_batch=data_batch)

        outputs = self.runner.model.test_step(data_batch)
        self.evaluator.process(outputs, data_batch, metrics)
        self.runner.call_hook(
            'after_test_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
