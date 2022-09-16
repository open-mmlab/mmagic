# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from copy import deepcopy
from functools import partial
from typing import Optional, Sequence

import torch
from mmengine.hooks import Hook
from mmengine.model.wrappers import is_model_wrapper
from mmengine.registry import HOOKS
from mmengine.runner import Runner
from mmengine.utils import is_tuple_of

DATA_BATCH = Optional[Sequence[dict]]


@HOOKS.register_module()
class ExponentialMovingAverageHook(Hook):
    """Exponential Moving Average Hook.

    Exponential moving average is a trick that widely used in current GAN
    literature, e.g., PGGAN, StyleGAN, and BigGAN. This general idea of it is
    maintaining a model with the same architecture, but its parameters are
    updated as a moving average of the trained weights in the original model.
    In general, the model with moving averaged weights achieves better
    performance.

    Args:
        module_keys (str | tuple[str]): The name of the ema model. Note that we
            require these keys are followed by '_ema' so that we can easily
            find the original model by discarding the last four characters.
        interp_mode (str, optional): Mode of the interpolation method.
            Defaults to 'lerp'.
        interp_cfg (dict | None, optional): Set arguments of the interpolation
            function. Defaults to None.
        interval (int, optional): Evaluation interval (by iterations).
            Default: -1.
        start_iter (int, optional): Start iteration for ema. If the start
            iteration is not reached, the weights of ema model will maintain
            the same as the original one. Otherwise, its parameters are updated
            as a moving average of the trained weights in the original model.
            Default: 0.
    """

    def __init__(self,
                 module_keys,
                 interp_mode='lerp',
                 interp_cfg=None,
                 interval=-1,
                 start_iter=0):
        super().__init__()
        assert isinstance(module_keys, str) or is_tuple_of(module_keys, str)
        self.module_keys = (module_keys, ) if isinstance(module_keys,
                                                         str) else module_keys
        # sanity check for the format of module keys
        for k in self.module_keys:
            assert k.endswith(
                '_ema'), 'You should give keys that end with "_ema".'
        self.interp_mode = interp_mode
        self.interp_cfg = dict() if interp_cfg is None else deepcopy(
            interp_cfg)
        self.interval = interval
        self.start_iter = start_iter

        assert hasattr(
            self, interp_mode
        ), f'Currently, we do not support {self.interp_mode} for EMA.'
        self.interp_func = partial(
            getattr(self, interp_mode), **self.interp_cfg)

    @staticmethod
    def lerp(a, b, momentum=0.999, momentum_nontrainable=0., trainable=True):
        """This is the function to perform linear interpolation between a and
        b.

        Args:
            a (float): number a
            b (float): bumber b
            momentum (float, optional): momentum. Defaults to 0.999.
            momentum_nontrainable (float, optional): Defaults to 0.
            trainable (bool, optional): trainable flag. Defaults to True.

        Returns:
            _type_: _description_
        """
        m = momentum if trainable else momentum_nontrainable
        return a + (b - a) * m

    def every_n_iters(self, runner: Runner, n: int):
        """This is the function to perform every n iterations.

        Args:
            runner (Runner): runner used to drive the whole pipeline
            n (int): the number of iterations

        Returns:
            int: the latest iterations
        """
        if runner.iter < self.start_iter:
            return True
        return (runner.iter + 1 - self.start_iter) % n == 0 if n > 0 else False

    @torch.no_grad()
    def after_train_iter(self,
                         runner: Runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
        """This is the function to perform after each training iteration.

        Args:
            runner (Runner): runner to drive the pipeline
            batch_idx (int): the id of batch
            data_batch (DATA_BATCH, optional): data batch. Defaults to None.
            outputs (Optional[dict], optional): output. Defaults to None.
        """

        if not self.every_n_iters(runner, self.interval):
            return

        model = runner.model.module if is_model_wrapper(
            runner.model) else runner.model

        for key in self.module_keys:
            # get current ema states
            ema_net = getattr(model, key)
            states_ema = ema_net.state_dict(keep_vars=False)
            # get currently original states
            net = getattr(model, key[:-4])
            states_orig = net.state_dict(keep_vars=True)

            for k, v in states_orig.items():
                if runner.iter < self.start_iter:
                    states_ema[k].data.copy_(v.data)
                else:
                    states_ema[k] = self.interp_func(
                        v, states_ema[k], trainable=v.requires_grad).detach()
            ema_net.load_state_dict(states_ema, strict=True)

    def before_run(self, runner: Runner):
        """This is the function perform before each run.

        Args:
            runner (Runner): runner used to drive the whole pipeline

        Raises:
            RuntimeError: error message
        """
        model = runner.model.module if is_model_wrapper(
            runner.model) else runner.model
        # sanity check for ema model
        for k in self.module_keys:
            if not hasattr(model, k) and not hasattr(model, k[:-4]):
                raise RuntimeError(
                    f'Cannot find both {k[:-4]} and {k} network for EMA hook.')
            if not hasattr(model, k) and hasattr(model, k[:-4]):
                setattr(model, k, deepcopy(getattr(model, k[:-4])))
                warnings.warn(
                    f'We do not suggest construct and initialize EMA model {k}'
                    ' in hook. You may explicitly define it by yourself.')
