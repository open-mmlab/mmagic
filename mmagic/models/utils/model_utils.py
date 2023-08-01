# Copyright (c) OpenMMLab. All rights reserved.
import logging
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from mmengine import print_log
from mmengine.model.weight_init import (constant_init, kaiming_init,
                                        normal_init, update_init_info,
                                        xavier_init)
from mmengine.registry import Registry
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm
from torch import Tensor
from torch.nn import init

from mmagic.structures import DataSample
from mmagic.utils.typing import ForwardInputs
from .tome_utils import (add_tome_cfg_hook, build_mmagic_tomesd_block,
                         build_mmagic_wrapper_tomesd_block, isinstance_str)


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
            init_info = (f'Initialize {m.__class__.__name__} by \'init_type\' '
                         f'{init_type}.')
        elif classname.find('BatchNorm2d') != -1:
            # BatchNorm Layer's weight is not a matrix;
            # only normal distribution applies.
            normal_init(m, 1.0, init_gain)
            init_info = (f'{m.__class__.__name__} is BatchNorm2d, initialize '
                         'by Norm initialization with mean=1, '
                         f'std={init_gain}')

        if hasattr(m, '_params_init_info'):
            update_init_info(module, init_info)

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
            'Input \'noise_size\' is inconsistent with '
            f'\'generator.noise_size\'. Receive \'{noise_size}\' and '
            f'\'{model_noise_size}\'.')
    else:
        noise_size = noise_size or model_noise_size

    return noise_size


def get_valid_num_batches(batch_inputs: Optional[ForwardInputs] = None,
                          data_samples: List[DataSample] = None) -> int:
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
    # attempt to infer num_batches from batch_inputs
    if batch_inputs is not None:
        if isinstance(batch_inputs, Tensor):
            return batch_inputs.shape[0]

        # get num_batches from batch_inputs
        num_batches_dict = {
            k: v.shape[0]
            for k, v in batch_inputs.items() if isinstance(v, Tensor)
        }
        if 'num_batches' in batch_inputs:
            num_batches_dict['num_batches'] = batch_inputs['num_batches']

        if num_batches_dict:
            num_batches_inputs = list(num_batches_dict.values())[0]
            # ensure all num_batches are same
            assert all([
                bz == num_batches_inputs for bz in num_batches_dict.values()
            ]), ('\'num_batches\' is inconsistency among the preprocessed '
                 f'input. \'num_batches\' parsed results: {num_batches_dict}')
        else:
            num_batches_inputs = None
    else:
        num_batches_inputs = None

    # attempt to infer num_batches from data_samples
    if data_samples is not None:
        num_batches_samples = len(data_samples)
    else:
        num_batches_samples = None

    if not (num_batches_inputs or num_batches_samples):
        print_log(
            'Cannot get \'num_batches\' from both \'inputs\' and '
            '\'data_samples\', automatically set \'num_batches\' as 1. '
            'This may leads to potential error.', 'current', logging.WARNING)
        return 1
    elif num_batches_inputs and num_batches_samples:
        assert num_batches_inputs == num_batches_samples, (
            '\'num_batches\' inferred from \'inputs\' and \'data_samples\' '
            f'are different, ({num_batches_inputs} vs. {num_batches_samples}).'
            ' Please check your input carefully.')

    return num_batches_inputs or num_batches_samples


def build_module(module: Union[dict, nn.Module], builder: Registry, *args,
                 **kwargs) -> Any:
    """Build module from config or return the module itself.

    Args:
        module (Union[dict, nn.Module]): The module to build.
        builder (Registry): The registry to build module.
        *args, **kwargs: Arguments passed to build function.

    Returns:
        Any: The built module.
    """
    if isinstance(module, dict):
        return builder.build(module, *args, **kwargs)
    elif isinstance(module, nn.Module):
        return module
    else:
        raise TypeError(
            f'Only support dict and nn.Module, but got {type(module)}.')


def xformers_is_enable(verbose: bool = False) -> bool:
    """Check whether xformers is installed.
    Args:
        verbose (bool): Whether to print the log.

    Returns:
        bool: Whether xformers is installed.
    """
    from mmagic.utils import try_import
    xformers = try_import('xformers')
    if xformers is None and verbose:
        print_log('Do not support Xformers.', 'current')
    return xformers is not None


def set_xformers(module: nn.Module, prefix: str = '') -> nn.Module:
    """Set xformers' efficient Attention for attention modules.

    Args:
        module (nn.Module): The module to set xformers.
        prefix (str): The prefix of the module name.

    Returns:
        nn.Module: The module with xformers' efficient Attention.
    """

    if not xformers_is_enable():
        print_log('Do not support Xformers. Please install Xformers first. '
                  'The program will run without Xformers.')
        return

    for n, m in module.named_children():
        if hasattr(m, 'set_use_memory_efficient_attention_xformers'):
            # set xformers for Diffusers' Cross Attention
            m.set_use_memory_efficient_attention_xformers(True)
            module_name = f'{prefix}.{n}' if prefix else n
            print_log(
                'Enable Xformers for HuggingFace Diffusers\' '
                f'module \'{module_name}\'.', 'current')
        else:
            set_xformers(m, prefix=n)

    return module


def set_tomesd(model: torch.nn.Module,
               ratio: float = 0.5,
               max_downsample: int = 1,
               sx: int = 2,
               sy: int = 2,
               use_rand: bool = True,
               merge_attn: bool = True,
               merge_crossattn: bool = False,
               merge_mlp: bool = False):
    """Patches a stable diffusion model with ToMe. Apply this to the highest
    level stable diffusion object.

    Refer to: https://github.com/dbolya/tomesd/blob/main/tomesd/patch.py#L173 # noqa

    Args:
        model (torch.nn.Module): A top level Stable Diffusion module to patch in place.
        ratio (float): The ratio of tokens to merge. I.e., 0.4 would reduce the total
            number of tokens by 40%.The maximum value for this is 1-(1/(`sx` * `sy`)). By default,
            the max ratio is 0.75 (usually <= 0.5 is recommended). Higher values result in more speed-up,
            but with more visual quality loss.
        max_downsample (int): Apply ToMe to layers with at most this amount of downsampling.
            E.g., 1 only applies to layers with no downsampling, while 8 applies to all layers.
            Should be chosen from [1, 2, 4, or 8]. 1 and 2 are recommended.
        sx, sy (int, int): The stride for computing dst sets. A higher stride means you can merge
            more tokens, default setting of (2, 2) works well in most cases.
            `sx` and `sy` do not need to divide image size.
        use_rand (bool): Whether or not to allow random perturbations when computing dst sets.
            By default: True, but if you're having weird artifacts you can try turning this off.
        merge_attn (bool): Whether or not to merge tokens for attention (recommended).
        merge_crossattn (bool): Whether or not to merge tokens for cross attention (not recommended).
        merge_mlp (bool): Whether or not to merge tokens for the mlp layers (particular not recommended).

    Returns:
        model (torch.nn.Module): Model patched by ToMe.
    """

    # Make sure the module is not currently patched
    remove_tomesd(model)

    is_mmagic = isinstance_str(model, 'StableDiffusion') or isinstance_str(
        model, 'BaseModel')

    if is_mmagic:
        # Supports "StableDiffusion.unet" and "unet"
        diffusion_model = model.unet if hasattr(model, 'unet') else model
        if isinstance_str(diffusion_model, 'DenoisingUnet'):
            is_wrapper = False
        else:
            is_wrapper = True
    else:
        if not hasattr(model, 'model') or not hasattr(model.model,
                                                      'diffusion_model'):
            # Provided model not supported
            print('Expected a Stable Diffusion / Latent Diffusion model.')
            raise RuntimeError('Provided model was not supported.')
        diffusion_model = model.model.diffusion_model
        # TODO: can support more diffusion models, like Stability AI
        is_wrapper = None

    diffusion_model._tome_info = {
        'size': None,
        'hooks': [],
        'args': {
            'ratio': ratio,
            'max_downsample': max_downsample,
            'sx': sx,
            'sy': sy,
            'use_rand': use_rand,
            'merge_attn': merge_attn,
            'merge_crossattn': merge_crossattn,
            'merge_mlp': merge_mlp
        }
    }
    add_tome_cfg_hook(diffusion_model)

    for _, module in diffusion_model.named_modules():
        if isinstance_str(module, 'BasicTransformerBlock'):
            # TODO: can support more stable diffusion based models
            if is_mmagic:
                if is_wrapper is None:
                    raise NotImplementedError(
                        'Specific ToMe block not implemented')
                elif not is_wrapper:
                    make_tome_block_fn = build_mmagic_tomesd_block
                elif is_wrapper:
                    make_tome_block_fn = build_mmagic_wrapper_tomesd_block
            else:
                raise TypeError(
                    'Currently `tome` only support *stable-diffusion* model!')
            module.__class__ = make_tome_block_fn(module.__class__)
            module._tome_info = diffusion_model._tome_info

    return model


def remove_tomesd(model: torch.nn.Module):
    """Removes a patch from a ToMe Diffusion module if it was already patched.

    Refer to: https://github.com/dbolya/tomesd/blob/main/tomesd/patch.py#L251 # noqa
    """
    # For mmagic Stable Diffusion models
    model = model.unet if hasattr(model, 'unet') else model

    for _, module in model.named_modules():
        if hasattr(module, '_tome_info'):
            for hook in module._tome_info['hooks']:
                hook.remove()
            module._tome_info['hooks'].clear()

        if module.__class__.__name__ == 'ToMeBlock':
            module.__class__ = module._parent

    return model
