# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence

from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.structures import BaseDataElement


@HOOKS.register_module()
class BasicVisualizationHook(Hook):
    """Basic hook that invoke visualizers during validation and test.

    Args:
        interval (int | dict): Visualization interval. Default: {}.
        on_train (bool): Whether to call hook during train. Default to False.
        on_val (bool): Whether to call hook during validation. Default to True.
        on_test (bool): Whether to call hook during test. Default to True.
    """
    priority = 'NORMAL'

    def __init__(self,
                 interval: dict = {},
                 on_train=False,
                 on_val=True,
                 on_test=True):
        self._interval = interval
        self._sample_counter = 0
        self._vis_dir = None
        self._on_train = on_train
        self._on_val = on_val
        self._on_test = on_test

    def _after_iter(
        self,
        runner,
        batch_idx: int,
        data_batch: Optional[Sequence[dict]],
        outputs: Optional[Sequence[BaseDataElement]],
        mode=None,
    ) -> None:
        """Show or Write the predicted results.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the test loop.
            data_batch (Sequence[dict], optional): Data
                from dataloader. Defaults to None.
            outputs (Sequence[BaseDataElement], optional): Outputs from model.
                Defaults to None.
        """
        if mode == 'train' and (not self._on_train):
            return
        elif mode == 'val' and (not self._on_val):
            return
        elif mode == 'test' and (not self._on_test):
            return

        if isinstance(self._interval, int):
            interval = self._interval
        else:
            interval = self._interval.get(mode, 1)

        if self.every_n_inner_iters(batch_idx, interval):
            for data_sample in outputs:
                runner.visualizer.add_datasample(
                    data_sample, step=f'{mode}_{runner.iter}')
