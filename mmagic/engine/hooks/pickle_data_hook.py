# Copyright (c) OpenMMLab. All rights reserved.

import logging
import os
import pickle
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from mmengine import is_list_of, mkdir_or_exist, print_log
from mmengine.dist import master_only
from mmengine.hooks import Hook
from mmengine.runner import Runner
from torch import Tensor

from mmagic.registry import HOOKS

DATA_BATCH = Optional[Sequence[dict]]


@HOOKS.register_module()
class PickleDataHook(Hook):
    """Pickle Useful Data Hook.

    This hook will be used in SinGAN training for saving some important data
    that will be used in testing or inference.

    Args:
        output_dir (str): The output path for saving pickled data.
        data_name_list (list[str]): The list contains the name of results in
            outputs dict.
        interval (int): The interval of calling this hook. If set to -1,
            the PickleDataHook will not be called during training. Default: -1.
        before_run (bool, optional): Whether to save before running.
            Defaults to False.
        after_run (bool, optional): Whether to save after running.
            Defaults to False.
        filename_tmpl (str, optional): Format string used to save images. The
            output file name will be formatted as this args.
            Defaults to 'iter_{}.pkl'.
    """

    def __init__(self,
                 output_dir,
                 data_name_list,
                 interval=-1,
                 before_run=False,
                 after_run=False,
                 filename_tmpl='iter_{}.pkl'):
        assert is_list_of(data_name_list, str)
        self.output_dir = output_dir
        self.data_name_list = data_name_list
        self.interval = interval
        self.filename_tmpl = filename_tmpl
        self._before_run = before_run
        self._after_run = after_run

    @master_only
    def after_run(self, runner):
        """The behavior after each train iteration.

        Args:
            runner (object): The runner.
        """
        if self._after_run:
            self._pickle_data(runner)

    @master_only
    def before_run(self, runner):
        """The behavior after each train iteration.

        Args:
            runner (object): The runner.
        """
        if self._before_run:
            self._pickle_data(runner)

    @master_only
    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None):
        """The behavior after each train iteration.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (Sequence[dict], optional): Data from dataloader.
                Defaults to None.
            outputs (dict, optional): Outputs from model.
                Defaults to None.
        """
        if not self.every_n_train_iters(runner, self.interval):
            return
        self._pickle_data(runner)

    def _pickle_data(self, runner: Runner):
        """Save target data to pickle file.

        Args:
            runner (Runner): The runner of the training process.
        """
        filename = self.filename_tmpl.format(runner.iter + 1)
        if not hasattr(self, '_out_dir'):
            self._out_dir = os.path.join(runner.work_dir, self.output_dir)
        mkdir_or_exist(self._out_dir)
        file_path = os.path.join(self._out_dir, filename)
        with open(file_path, 'wb') as f:
            module = runner.model
            if hasattr(module, 'module'):
                module = module.module
            not_find_keys = []
            data_dict = {}
            for k in self.data_name_list:
                if hasattr(module, k):
                    data_dict[k] = self._get_numpy_data(getattr(module, k))
                else:
                    not_find_keys.append(k)
            pickle.dump(data_dict, f)
            print_log(f'Pickle data in {filename}', 'current')

            if len(not_find_keys) > 0:
                print_log(
                    f'Cannot find keys for pickling: {not_find_keys}',
                    'current',
                    level=logging.WARN)
            f.flush()

    def _get_numpy_data(
        self, data: Tuple[List[Tensor], Tensor, int]
    ) -> Tuple[List[np.ndarray], np.ndarray, int]:
        """Convert tensor or list of tensor to numpy or list of numpy.

        Args:
            data (Tuple[List[Tensor], Tensor, int]): Data to be converted.

        Returns:
            Tuple[List[np.ndarray], np.ndarray, int]: Converted data.
        """
        if isinstance(data, list):
            return [self._get_numpy_data(x) for x in data]

        if isinstance(data, torch.Tensor):
            return data.cpu().numpy()

        return data
