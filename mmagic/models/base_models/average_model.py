# Copyright (c) OpenMMLab. All rights reserved.
import itertools
import warnings
from typing import List, Optional

import torch
import torch.nn as nn
from mmengine.model import BaseAveragedModel
from torch import Tensor

from mmagic.registry import MODELS

# NOTICE: Since mmengine do not support loading ``state_dict`` without wrap
# ema module with ``BaseAveragedModel`` currently, we rewrite
# ``ExponentialMovingAverage`` and add ``_load_from_state_dict`` temporarily


@MODELS.register_module()
class ExponentialMovingAverage(BaseAveragedModel):
    r"""Implements the exponential moving average (EMA) of the model.

    All parameters are updated by the formula as below:

        .. math::

            Xema_{t+1} = (1 - momentum) * Xema_{t} +  momentum * X_t

    Args:
        model (nn.Module): The model to be averaged.
        momentum (float): The momentum used for updating ema parameter.
            Defaults to 0.0002.
            Ema's parameter are updated with the formula
            :math:`averaged\_param = (1-momentum) * averaged\_param +
            momentum * source\_param`.
        interval (int): Interval between two updates. Defaults to 1.
        device (torch.device, optional): If provided, the averaged model will
            be stored on the :attr:`device`. Defaults to None.
        update_buffers (bool): if True, it will compute running averages for
            both the parameters and the buffers of the model. Defaults to
            False.
    """  # noqa: W605

    def __init__(self,
                 model: nn.Module,
                 momentum: float = 0.0002,
                 interval: int = 1,
                 device: Optional[torch.device] = None,
                 update_buffers: bool = False) -> None:
        super().__init__(model, interval, device, update_buffers)
        assert 0.0 < momentum < 1.0, 'momentum must be in range (0.0, 1.0)'\
                                     f'but got {momentum}'
        if momentum > 0.5:
            warnings.warn(
                'The value of momentum in EMA is usually a small number,'
                'which is different from the conventional notion of '
                f'momentum but got {momentum}. Please make sure the '
                f'value is correct.')
        self.momentum = momentum

    def avg_func(self, averaged_param: Tensor, source_param: Tensor,
                 steps: int) -> None:
        """Compute the moving average of the parameters using exponential
        moving average.

        Args:
            averaged_param (Tensor): The averaged parameters.
            source_param (Tensor): The source parameters.
            steps (int): The number of times the parameters have been
                updated.
        """
        averaged_param.mul_(1 - self.momentum).add_(
            source_param, alpha=self.momentum)

    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: list, unexpected_keys: list,
                              error_msgs: List[str]) -> None:
        """Overrides ``nn.Module._load_from_state_dict`` to support loading
        ``state_dict`` without wrap ema module with ``BaseAveragedModel``.

        In OpenMMLab 1.0, model will not wrap ema submodule with
        ``BaseAveragedModel``, and the ema weight key in `state_dict` will
        miss `module` prefix. Therefore, ``BaseAveragedModel`` need to
        automatically add the ``module`` prefix if the corresponding key in
        ``state_dict`` misses it.

        Args:
            state_dict (dict): A dict containing parameters and
                persistent buffers.
            prefix (str): The prefix for parameters and buffers used in this
                module
            local_metadata (dict): a dict containing the metadata for this
                module.
            strict (bool): Whether to strictly enforce that the keys in
                :attr:`state_dict` with :attr:`prefix` match the names of
                parameters and buffers in this module
            missing_keys (List[str]): if ``strict=True``, add missing keys to
                this list
            unexpected_keys (List[str]): if ``strict=True``, add unexpected
                keys to this list
            error_msgs (List[str]): error messages should be added to this
                list, and will be reported together in
                :meth:`~torch.nn.Module.load_state_dict`.
        """

        for key, value in list(state_dict.items()):
            # To support load the pretrained model, which does not wrap ema
            # module with `BaseAveragedModel`, `BaseAveragedModel` will
            # automatically add `module` prefix to the `state_dict` which
            # key starts with the custom prefix. For example, the old
            # checkpoint with `state_dict` with keys:
            # ['layer.weight', 'layer.bias', 'ema.steps', 'ema.weight', 'ema.bias'] # noqa: E501
            # will be replaced with:
            # ['layer.weight', 'layer.bias', 'ema.steps', 'ema.module.weight', 'ema.module.bias'] # noqa: E501

            # The key added with `module` prefix needs to satisfy
            # three conditions.
            # 1. key starts with current prefix, such as `model.ema`.
            # 2. The content after the prefix does not start with the `module`
            # 3. Key does not end with steps.
            if key.startswith(prefix) and not key[len(prefix):].startswith(
                    'module') and not key.endswith('steps'):
                new_key = key[:len(prefix)] + 'module.' + key[len(prefix):]
                state_dict[new_key] = value
                state_dict.pop(key)
        state_dict.setdefault(prefix + 'steps', torch.tensor(0))
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    def sync_buffers(self, model: nn.Module) -> None:
        """Copy buffer from model to averaged model.

        Args:
            model (nn.Module): The model whose parameters will be averaged.
        """
        # if not update buffer, copy buffer from orig model
        if self.update_buffers:
            warnings.warn(
                '`update_buffers` is set to True in this ema model, and '
                'buffers will be updated in `update_parameters`.')

        avg_buffer = itertools.chain(self.module.buffers())
        orig_buffer = itertools.chain(model.buffers())
        for b_avg, b_orig in zip(avg_buffer, orig_buffer):
            b_avg.data.copy_(b_orig.data)

    def sync_parameters(self, model: nn.Module) -> None:
        """Copy buffer and parameters from model to averaged model.

        Args:
            model (nn.Module): The model whose parameters will be averaged.
        """
        # before ema, copy weights from orig
        avg_param = (
            itertools.chain(self.module.parameters(), self.module.buffers()))
        src_param = (itertools.chain(model.parameters(), model.buffers()))
        for p_avg, p_src in zip(avg_param, src_param):
            p_avg.data.copy_(p_src.data)


@MODELS.register_module()
class RampUpEMA(BaseAveragedModel):
    r"""Implements the exponential moving average with ramping up momentum.

    Ref: https://github.com/NVlabs/stylegan3/blob/master/training/training_loop.py # noqa

    Args:
        model (nn.Module): The model to be averaged.
        interval (int): Interval between two updates. Defaults to 1.
        ema_kimg (int, optional): EMA kimgs. Defaults to 10.
        ema_rampup (float, optional): Ramp up rate. Defaults to 0.05.
        batch_size (int, optional): Global batch size. Defaults to 32.
        eps (float, optional): Ramp up epsilon. Defaults to 1e-8.
        start_iter (int, optional): EMA start iter. Defaults to 0.
        device (torch.device, optional): If provided, the averaged model will
            be stored on the :attr:`device`. Defaults to None.
        update_buffers (bool): if True, it will compute running averages for
            both the parameters and the buffers of the model. Defaults to
            False.
    """  # noqa: W605

    def __init__(self,
                 model: nn.Module,
                 interval: int = 1,
                 ema_kimg: int = 10,
                 ema_rampup: float = 0.05,
                 batch_size: int = 32,
                 eps: float = 1e-8,
                 start_iter: int = 0,
                 device: Optional[torch.device] = None,
                 update_buffers: bool = False) -> None:
        """_summary_"""
        super().__init__(model, interval, device, update_buffers)
        self.interval = interval
        self.ema_kimg = ema_kimg
        self.ema_rampup = ema_rampup
        self.batch_size = batch_size
        self.eps = eps

    @staticmethod
    def rampup(steps, ema_kimg=10, ema_rampup=0.05, batch_size=4, eps=1e-8):
        """Ramp up ema momentum.

        Ref: https://github.com/NVlabs/stylegan3/blob/a5a69f58294509598714d1e88c9646c3d7c6ec94/training/training_loop.py#L300-L308 # noqa

        Args:
            steps:
            ema_kimg (int, optional): Half-life of the exponential moving
                average of generator weights. Defaults to 10.
            ema_rampup (float, optional): EMA ramp-up coefficient.If set to
                None, then rampup will be disabled. Defaults to 0.05.
            batch_size (int, optional): Total batch size for one training
                iteration. Defaults to 4.
            eps (float, optional): Epsiolon to avoid ``batch_size`` divided by
                zero. Defaults to 1e-8.

        Returns:
            dict: Updated momentum.
        """
        cur_nimg = (steps + 1) * batch_size
        ema_nimg = ema_kimg * 1000
        if ema_rampup is not None:
            ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
        ema_beta = 0.5**(batch_size / max(ema_nimg, eps))
        return ema_beta

    def avg_func(self, averaged_param: Tensor, source_param: Tensor,
                 steps: int) -> None:
        """Compute the moving average of the parameters using exponential
        moving average.

        Args:
            averaged_param (Tensor): The averaged parameters.
            source_param (Tensor): The source parameters.
            steps (int): The number of times the parameters have been
                updated.
        """
        momentum = 1. - self.rampup(self.steps, self.ema_kimg, self.ema_rampup,
                                    self.batch_size, self.eps)
        if not (0.0 < momentum < 1.0):
            warnings.warn('RampUp momentum must be in range (0.0, 1.0)'
                          f'but got {momentum}')
        averaged_param.mul_(1 - momentum).add_(source_param, alpha=momentum)

    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: list, unexpected_keys: list,
                              error_msgs: List[str]) -> None:
        """Overrides ``nn.Module._load_from_state_dict`` to support loading
        ``state_dict`` without wrap ema module with ``BaseAveragedModel``.

        In OpenMMLab 1.0, model will not wrap ema submodule with
        ``BaseAveragedModel``, and the ema weight key in `state_dict` will
        miss `module` prefix. Therefore, ``BaseAveragedModel`` need to
        automatically add the ``module`` prefix if the corresponding key in
        ``state_dict`` misses it.

        Args:
            state_dict (dict): A dict containing parameters and
                persistent buffers.
            prefix (str): The prefix for parameters and buffers used in this
                module
            local_metadata (dict): a dict containing the metadata for this
                module.
            strict (bool): Whether to strictly enforce that the keys in
                :attr:`state_dict` with :attr:`prefix` match the names of
                parameters and buffers in this module
            missing_keys (List[str]): if ``strict=True``, add missing keys to
                this list
            unexpected_keys (List[str]): if ``strict=True``, add unexpected
                keys to this list
            error_msgs (List[str]): error messages should be added to this
                list, and will be reported together in
                :meth:`~torch.nn.Module.load_state_dict`.
        """

        for key, value in list(state_dict.items()):
            # To support load the pretrained model, which does not wrap ema
            # module with `BaseAveragedModel`, `BaseAveragedModel` will
            # automatically add `module` prefix to the `state_dict` which
            # key starts with the custom prefix. For example, the old
            # checkpoint with `state_dict` with keys:
            # ['layer.weight', 'layer.bias', 'ema.steps', 'ema.weight', 'ema.bias'] # noqa: E501
            # will be replaced with:
            # ['layer.weight', 'layer.bias', 'ema.steps', 'ema.module.weight', 'ema.module.bias'] # noqa: E501

            # The key added with `module` prefix needs to satisfy
            # three conditions.
            # 1. key starts with current prefix, such as `model.ema`.
            # 2. The content after the prefix does not start with the `module`
            # 3. Key does not end with steps.
            if key.startswith(prefix) and not key[len(prefix):].startswith(
                    'module') and not key.endswith('steps'):
                new_key = key[:len(prefix)] + 'module.' + key[len(prefix):]
                state_dict[new_key] = value
                state_dict.pop(key)
        state_dict.setdefault(prefix + 'steps', torch.tensor(0))
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    def sync_buffers(self, model: nn.Module) -> None:
        """Copy buffer from model to averaged model.

        Args:
            model (nn.Module): The model whose parameters will be averaged.
        """
        # if not update buffer, copy buffer from orig model
        if self.update_buffers:
            warnings.warn(
                '`update_buffers` is set to True in this ema model, and '
                'buffers will be updated in `update_parameters`.')

        avg_buffer = itertools.chain(self.module.buffers())
        orig_buffer = itertools.chain(model.buffers())
        for b_avg, b_orig in zip(avg_buffer, orig_buffer):
            b_avg.data.copy_(b_orig.data)

    def sync_parameters(self, model: nn.Module) -> None:
        """Copy buffer and parameters from model to averaged model.

        Args:
            model (nn.Module): The model whose parameters will be averaged.
        """
        # before ema, copy weights from orig
        avg_param = (
            itertools.chain(self.module.parameters(), self.module.buffers()))
        src_param = (itertools.chain(model.parameters(), model.buffers()))
        for p_avg, p_src in zip(avg_param, src_param):
            p_avg.data.copy_(p_src.data)
