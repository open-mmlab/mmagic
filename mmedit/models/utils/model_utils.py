# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from mmengine.model.weight_init import (constant_init, kaiming_init,
                                        normal_init, xavier_init)
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm
from torch import Tensor
from torch.nn import init

from mmedit.utils.typing import ForwardInputs


def default_init_weights(module, scale=1):
    """Initialize network weights.

    Args:
        modules (nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale
        elif isinstance(m, nn.Linear):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale
        elif isinstance(m, _BatchNorm):
            constant_init(m.weight, val=1, bias=0)


def make_layer(block, num_blocks, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        block (nn.module): nn.module class for basic block.
        num_blocks (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_blocks):
        layers.append(block(**kwarg))
    return nn.Sequential(*layers)


def get_module_device(module):
    """Get the device of a module.

    Args:
        module (nn.Module): A module contains the parameters.

    Returns:
        torch.device: The device of the module.
    """
    try:
        next(module.parameters())
    except StopIteration:
        raise ValueError('The input module should contain parameters.')

    if next(module.parameters()).is_cuda:
        return next(module.parameters()).get_device()
    else:
        return torch.device('cpu')


def set_requires_grad(nets, requires_grad=False):
    """Set requires_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def generation_init_weights(module, init_type='normal', init_gain=0.02):
    """Default initialization of network weights for image generation.

    By default, we use normal init, but xavier and kaiming might work
    better for some applications.

    Args:
        module (nn.Module): Module to be initialized.
        init_type (str): The name of an initialization method:
            normal | xavier | kaiming | orthogonal. Default: 'normal'.
        init_gain (float): Scaling factor for normal, xavier and
            orthogonal. Default: 0.02.
    """

    def init_func(m):
        """Initialization function.

        Args:
            m (nn.Module): Module to be initialized.
        """
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                     or classname.find('Linear') != -1):
            if init_type == 'normal':
                normal_init(m, 0.0, init_gain)
            elif init_type == 'xavier':
                xavier_init(m, gain=init_gain, distribution='normal')
            elif init_type == 'kaiming':
                kaiming_init(
                    m,
                    a=0,
                    mode='fan_in',
                    nonlinearity='leaky_relu',
                    distribution='normal')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight, gain=init_gain)
                init.constant_(m.bias.data, 0.0)
            else:
                raise NotImplementedError(
                    f"Initialization method '{init_type}' is not implemented")
        elif classname.find('BatchNorm2d') != -1:
            # BatchNorm Layer's weight is not a matrix;
            # only normal distribution applies.
            normal_init(m, 1.0, init_gain)

    module.apply(init_func)


def get_valid_noise_size(noise_size: Optional[int],
                         generator: Union[Dict, nn.Module]) -> Optional[int]:
    """Get the value of `noise_size` from input, `generator` and check the
    consistency of these values. If no conflict is found, return that value.

    Args:
        noise_size (Optional[int]): `noise_size` passed to
            `BaseGAN_refactor`'s initialize function.
        generator (ModelType): The config or the model of generator.

    Returns:
        int | None: The noise size feed to generator.
    """
    if isinstance(generator, dict):
        model_noise_size = generator.get('noise_size', None)
    else:
        model_noise_size = getattr(generator, 'noise_size', None)

    # get noise_size
    if noise_size is not None and model_noise_size is not None:
        assert noise_size == model_noise_size, (
            'Input \'noise_size\' is unconsistency with '
            f'\'generator.noise_size\'. Receive \'{noise_size}\' and '
            f'\'{model_noise_size}\'.')
    else:
        noise_size = noise_size or model_noise_size

    return noise_size


def get_valid_num_batches(batch_inputs: ForwardInputs) -> int:
    """Try get the valid batch size from inputs.

    - If some values in `batch_inputs` are `Tensor` and 'num_batches' is in
      `batch_inputs`, we check whether the value of 'num_batches' and the the
      length of first dimension of all tensors are same. If the values are not
      same, `AssertionError` will be raised. If all values are the same,
      return the value.
    - If no values in `batch_inputs` is `Tensor`, 'num_batches' must be
      contained in `batch_inputs`. And this value will be returned.
    - If some values are `Tensor` and 'num_batches' is not contained in
      `batch_inputs`, we check whether all tensor have the same length on the
      first dimension. If the length are not same, `AssertionError` will be
      raised. If all length are the same, return the length as batch size.
    - If batch_inputs is a `Tensor`, directly return the length of the first
      dimension as batch size.

    Args:
        batch_inputs (ForwardInputs): Inputs passed to :meth:`forward`.

    Returns:
        int: The batch size of samples to generate.
    """
    if isinstance(batch_inputs, Tensor):
        return batch_inputs.shape[0]

    # get num_batces from batch_inputs
    num_batches_dict = {
        k: v.shape[0]
        for k, v in batch_inputs.items() if isinstance(v, Tensor)
    }
    if 'num_batches' in batch_inputs:
        num_batches_dict['num_batches'] = batch_inputs['num_batches']

    # ensure num_batches is not None
    assert len(num_batches_dict.keys()) > 0, (
        'Cannot get \'num_batches\' form preprocessed input '
        f'(\'{batch_inputs}\').')

    # ensure all num_batches are same
    num_batches = list(num_batches_dict.values())[0]
    assert all([
        bz == num_batches for bz in num_batches_dict.values()
    ]), ('\'num_batches\' is inconsistency among the preprocessed input. '
         f'\'num_batches\' parsed resutls: {num_batches_dict}')

    return num_batches
