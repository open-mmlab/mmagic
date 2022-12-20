# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from mmcv.runner import Hook
from torch.utils.data import DataLoader


class EvalIterHook(Hook):
    """Non-Distributed evaluation hook for iteration-based runner.

    This hook will regularly perform evaluation in a given interval when
    performing in non-distributed environment.

    Args:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval. Default: 1.
        eval_kwargs (dict): Other eval kwargs. It contains:
            save_image (bool): Whether to save image.
            save_path (str): The path to save image.
    """

    def __init__(self, dataloader, interval=1, **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError('dataloader must be a pytorch DataLoader, '
                            f'but got { type(dataloader)}')
        self.dataloader = dataloader
        self.interval = interval
        self.eval_kwargs = eval_kwargs
        self.save_image = self.eval_kwargs.pop('save_image', False)
        self.save_path = self.eval_kwargs.pop('save_path', None)

    def after_train_iter(self, runner):
        """The behavior after each train iteration.

        Args:
            runner (``mmcv.runner.BaseRunner``): The runner.
        """
        if not self.every_n_iters(runner, self.interval):
            return
        runner.log_buffer.clear()
        from mmedit.apis import single_gpu_test
        results = single_gpu_test(
            runner.model,
            self.dataloader,
            save_image=self.save_image,
            save_path=self.save_path,
            iteration=runner.iter)
        self.evaluate(runner, results)

    def evaluate(self, runner, results):
        """Evaluation function.

        Args:
            runner (``mmcv.runner.BaseRunner``): The runner.
            results (dict): Model forward results.
        """
        eval_res = self.dataloader.dataset.evaluate(
            results, logger=runner.logger, **self.eval_kwargs)
        for name, val in eval_res.items():
            if isinstance(val, dict):
                runner.log_buffer.output.update(val)
                continue
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True
        # call `after_val_epoch` after evaluation.
        # This is a hack.
        # Because epoch does not naturally exist In IterBasedRunner,
        # thus we consider the end of an evluation as the end of an epoch.
        # With this hack , we can support epoch based hooks.
        if 'iter' in runner.__class__.__name__.lower():
            runner.call_hook('after_val_epoch')


class DistEvalIterHook(EvalIterHook):
    """Distributed evaluation hook.

    Args:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval. Default: 1.
        tmpdir (str | None): Temporary directory to save the results of all
            processes. Default: None.
        gpu_collect (bool): Whether to use gpu or cpu to collect results.
            Default: False.
        eval_kwargs (dict): Other eval kwargs. It may contain:
            save_image (bool): Whether save image.
            save_path (str): The path to save image.
    """

    def __init__(self,
                 dataloader,
                 interval=1,
                 gpu_collect=False,
                 **eval_kwargs):
        super().__init__(dataloader, interval, **eval_kwargs)
        self.gpu_collect = gpu_collect

    def after_train_iter(self, runner):
        """The behavior after each train iteration.

        Args:
            runner (``mmcv.runner.BaseRunner``): The runner.
        """
        if not self.every_n_iters(runner, self.interval):
            return
        runner.log_buffer.clear()
        from mmedit.apis import multi_gpu_test
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect,
            save_image=self.save_image,
            save_path=self.save_path,
            iteration=runner.iter)
        if runner.rank == 0:
            print('\n')
            self.evaluate(runner, results)
